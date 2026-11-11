#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Fused op chains; each kernel computes the same expressions as the chain of standard ops it replaces.
// Reductions use a work-group of FUSED_WG items (FUSED_WG_ROW for the per-token row kernels).

#define FUSED_MAX_HC  8

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
