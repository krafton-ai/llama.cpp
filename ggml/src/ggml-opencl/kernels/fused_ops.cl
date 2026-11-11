#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Fused op chains; each kernel computes the same expressions as the chain of standard ops it replaces.
// Reductions use a work-group of FUSED_WG items (FUSED_WG_ROW for the per-token row kernels).

#define FUSED_WG      256
#define FUSED_MAX_HC  8

#if defined(cl_khr_subgroups) || defined(__opencl_c_subgroups)
#define FUSED_SUBGROUPS 1
#endif
#if defined(cl_khr_subgroups)
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

// sum of v over the work-group (a power of two, at most FUSED_WG_ROW items); every item gets the result.
// With subgroups: one reduce per subgroup, then a sum of the subgroup results (2 barriers in all).
static inline float fused_wg_sum(float v, local float * tmp) {
#ifdef FUSED_SUBGROUPS
    const float sv = sub_group_reduce_add(v);
    const uint nsg = get_num_sub_groups();
    if (get_sub_group_local_id() == 0) {
        tmp[get_sub_group_id()] = sv;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float r = 0.0f;
    for (uint k = 0; k < nsg; ++k) {
        r += tmp[k];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
#else
    const int lid = get_local_id(0);
    tmp[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = get_local_size(0) / 2; s > 0; s >>= 1) {
        if (lid < s) {
            tmp[lid] += tmp[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const float r = tmp[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return r;
#endif
}

//------------------------------------------------------------------------------
// rms_norm_mul_mat_add_part: dst [nmix + 1, nsplit, nt]; group (k, s, t): fn row k . x chunk s of token t,
// k == nmix: sum x^2 of the chunk
//------------------------------------------------------------------------------
#define RMS_NORM_MUL_MAT_PART_BODY(LOAD_W)                                                   \
    const int k = get_group_id(0), s = get_group_id(1), t = get_group_id(2);                 \
    const int lid = get_local_id(0);                                                         \
    global const float * x = (global const float *) (src0 + offset0 + t * nb01) + s * chunk; \
    float acc = 0.0f;                                                                        \
    if (k == nmix) {                                                                         \
        for (int i = lid; i < chunk; i += FUSED_WG) {                                        \
            acc += x[i] * x[i];                                                              \
        }                                                                                    \
    } else {                                                                                 \
        for (int i = lid; i < chunk; i += FUSED_WG) {                                        \
            acc += (LOAD_W) * x[i];                                                          \
        }                                                                                    \
    }                                                                                        \
    acc = fused_wg_sum(acc, tmp);                                                            \
    if (lid == 0) {                                                                          \
        *(global float *) (dst + offsetd + k * sizeof(float) + s * nb1 + t * nb2) = acc;     \
    }

kernel void kernel_rms_norm_mul_mat_add_part_f16(
        global const char * src0,
        ulong offset0,
        ulong nb01,
        global const char * src1,
        ulong offset1,
        ulong nb11,
        global char * dst,
        ulong offsetd,
        ulong nb1,
        ulong nb2,
        int nmix,
        int chunk) {
    local float tmp[FUSED_WG];
    global const half * w = (global const half *) (src1 + offset1 + get_group_id(0) * nb11) + get_group_id(1) * chunk;
    RMS_NORM_MUL_MAT_PART_BODY(vload_half(i, w))
}

kernel void kernel_rms_norm_mul_mat_add_part_f32(
        global const char * src0,
        ulong offset0,
        ulong nb01,
        global const char * src1,
        ulong offset1,
        ulong nb11,
        global char * dst,
        ulong offsetd,
        ulong nb1,
        ulong nb2,
        int nmix,
        int chunk) {
    local float tmp[FUSED_WG];
    global const float * w = (global const float *) (src1 + offset1 + get_group_id(0) * nb11) + get_group_id(1) * chunk;
    RMS_NORM_MUL_MAT_PART_BODY(w[i])
}

//------------------------------------------------------------------------------
// rms_norm_mul_mat_add_reduce: second half of RMS_NORM -> MUL_MAT -> ADD(base).
//   parts [nmix + 1, nsplit, nt] from kernel_rms_norm_mul_mat_add_part_*: rows 0..nmix-1 partial w.x, row nmix partial x.x
//   y[k, t] = sum_s parts[k, s, t] / sqrt(sum_s parts[nmix, s, t] / n + eps) + base[k]
//------------------------------------------------------------------------------
kernel void kernel_rms_norm_mul_mat_add_reduce(
        global const char * parts,
        ulong offset_p,
        ulong nbp1,
        ulong nbp2,
        global const float * base,
        ulong offset_b,
        global char * dst,
        ulong offsetd,
        ulong nb1,
        int nmix,
        int nsplit,
        int n,
        float eps,
        int nt) {
    const int k = get_global_id(0);
    const int t = get_global_id(1);
    if (k >= nmix || t >= nt) {
        return;
    }
    base = (global const float *) ((global const char *) base + offset_b);
    float acc = 0.0f, ss = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        global const float * p = (global const float *) (parts + offset_p + s * nbp1 + t * nbp2);
        acc += p[k];
        ss += p[nmix];
    }
    *(global float *) (dst + offsetd + t * nb1 + k * sizeof(float)) = acc / sqrt(ss / n + eps) + base[k];
}

//------------------------------------------------------------------------------
// dsv4_hc_post_sum (DSV4_HC_POST with up to FUSED_MAX_HC streams): item per (i, t)
//   y[i, d, t] = x[i, t]*post[d, t] + sum_s res[i, s, t]*comb[d, s, t]   (has_comb == 0: + res[i, d, t])
//   has_sum: y[i, hc, t] = sum_d y[i, d, t]
//------------------------------------------------------------------------------
kernel void kernel_dsv4_hc_post_sum(
        global const char * x,
        ulong offset_x,
        ulong nbx0,
        ulong nbx1,
        global const char * res,
        ulong offset_r,
        ulong nbr0,
        ulong nbr1,
        ulong nbr2,
        global const char * post,
        ulong offset_p,
        ulong nbp0,
        ulong nbp1,
        global const char * comb,
        ulong offset_c,
        ulong nbc0,
        ulong nbc1,
        ulong nbc2,
        global char * dst,
        ulong offsetd,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        int n,
        int nt,
        int hc,
        int has_comb,
        int has_sum,
        global char * sdst,
        ulong offset_sd,
        ulong nbs1) {
    const int i = get_global_id(0);
    const int t = get_global_id(1);
    if (i >= n || t >= nt) {
        return;
    }
    float rv[FUSED_MAX_HC];
    for (int s = 0; s < hc; ++s) {
        rv[s] = *(global const float *) (res + offset_r + i * nbr0 + s * nbr1 + t * nbr2);
    }
    const float xv = *(global const float *) (x + offset_x + i * nbx0 + t * nbx1);
    float tot = 0.0f;
    for (int d = 0; d < hc; ++d) {
        float v = xv * *(global const float *) (post + offset_p + d * nbp0 + t * nbp1);
        if (has_comb) {
            for (int s = 0; s < hc; ++s) {
                v += rv[s] * *(global const float *) (comb + offset_c + d * nbc0 + s * nbc1 + t * nbc2);
            }
        } else {
            v += rv[d];
        }
        *(global float *) (dst + offsetd + i * nb0 + d * nb1 + t * nb2) = v;
        tot += v;
    }
    if (has_sum) {
        *(global float *) (sdst + offset_sd + i * sizeof(float) + t * nbs1) = tot;
    }
}
