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
// group size of mul_mat_pool_max_div_add: nmix and hc*hc <= MUL_MAT_POOL_WG
#define MUL_MAT_POOL_WG 64
//------------------------------------------------------------------------------
// mul_mat_pool_max_div_add: y [hc*hc, t] = c0 + e/rho, e = K m, rho = 1 + max_o e[o]*r[o], m = mixes [nmix, t]
//------------------------------------------------------------------------------
kernel void kernel_mul_mat_pool_max_div_add(
        global const char * src0,
        ulong offset0,
        ulong nb01,
        global const float * chart,
        ulong offset_c,
        global char * dst,
        ulong offsetd,
        ulong nb1,
        int hc,
        int nmix,
        int nt) {
    local float m[MUL_MAT_POOL_WG];
    local float er[MUL_MAT_POOL_WG];
    const int t = get_group_id(0);
    const int lid = get_local_id(0);
    chart = (global const float *) ((global const char *) chart + offset_c);
    const int hc2 = hc * hc;
    global const float * K = chart;
    global const float * r = chart + nmix * hc2;
    global const float * c0 = r + hc2;
    global float * y = (global float *) (dst + offsetd + t * nb1);
    if (lid < nmix) {
        m[lid] = ((global const float *) (src0 + offset0 + t * nb01))[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float e = 0.0f;
    if (lid < hc2) {
        for (int k = 0; k < nmix; ++k) {
            e += K[lid * nmix + k] * m[k];
        }
        er[lid] = e * r[lid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < hc2) {
        float mx = er[0];
        for (int o = 1; o < hc2; ++o) {
            mx = fmax(mx, er[o]);
        }
        y[lid] = c0[lid] + e / (1.0f + mx);
    }
}

//------------------------------------------------------------------------------
// sigmoid_div_mul: item per (j = h*hd + i, t)
//   beta[h, t] = sigmoid(fab[t, rank + h]),  w = sigmoid(fab[t, rank + nh + h]),  v'[i, h, t] = v[i, h, t] * (w / beta)
// same expressions as the standard SIGMOID, DIV and MUL kernels
//------------------------------------------------------------------------------
kernel void kernel_sigmoid_div_mul(
        global const char * fab,
        ulong offset_a,
        ulong nba1,
        global const char * v,
        ulong offset_v,
        ulong nbv1,
        ulong nbv2,
        global char * vout,
        ulong offset_vo,
        global char * bout,
        ulong offset_bo,
        int rank,
        int hd,
        int nh,
        int nt) {
    const int j = get_global_id(0);
    const int t = get_global_id(1);
    if (j >= hd * nh || t >= nt) {
        return;
    }
    const int h = j / hd, i = j % hd;
    global const float * a = (global const float *) (fab + offset_a + t * nba1);
    const float beta = 1.0f / (1.0f + exp(-a[rank + h]));
    const float w = 1.0f / (1.0f + exp(-a[rank + nh + h]));
    const float ratio = w / beta;
    const float vv = *(global const float *) (v + offset_v + i * sizeof(float) + h * nbv1 + t * nbv2);
    ((global float *) (vout + offset_vo))[(t * nh + h) * hd + i] = vv * ratio;
    if (i == 0) {
        ((global float *) (bout + offset_bo))[t * nh + h] = beta;
    }
}

//------------------------------------------------------------------------------
// add_sigmoid_scale: ADD(bias row) -> SIGMOID -> SCALE, y = sigmoid(x + b[j]) * s + c
//------------------------------------------------------------------------------
kernel void kernel_add_sigmoid_scale(
        global const char * x,
        ulong offset_x,
        ulong nbx1,
        global const float * b,
        ulong offset_b,
        global char * y,
        ulong offset_y,
        int n,
        int nr,
        float s,
        float c) {
    const int j = get_global_id(0);
    const int r = get_global_id(1);
    if (j >= n || r >= nr) {
        return;
    }
    b = (global const float *) ((global const char *) b + offset_b);
    const float z = *(global const float *) (x + offset_x + r * nbx1 + j * sizeof(float)) + b[j];
    ((global float *) (y + offset_y))[r * n + j] = (1.0f / (1.0f + exp(-z))) * s + c;
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
