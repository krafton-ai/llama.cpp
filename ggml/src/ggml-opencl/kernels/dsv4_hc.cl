// DeepSeek V4 style hyper-connection ops (mHC): DSV4_HC_PRE, DSV4_HC_POST, DSV4_HC_COMB.
// Same math and summation order as the CPU reference (ggml-cpu/ops.cpp) and the CUDA kernels (ggml-cuda/dsv4-hc.cu).
// The host builds this program without -cl-fast-relaxed-math / -cl-mad-enable; products are added with an
// explicit fma() in the same order as the reference loops (the CPU and CUDA compilers contract those to FMA too).

#pragma OPENCL FP_CONTRACT OFF

#define DSV4_HC 4

// result[i, t] = scale * sum_h x[i, h, t] * w[h, t]
// gated:  w[i, h, t] = sigmoid(gate[i, h, t])
// strides in bytes; one work-item per (i, t)
kernel void kernel_dsv4_hc_pre_f32(
        global const char * x,
        ulong offx,
        global const char * w,
        ulong offw,
        global char * dst,
        ulong offd,
        int n_embd,
        int hc,
        int n_tokens,
        ulong nbx0,
        ulong nbx1,
        ulong nbx2,
        ulong nbw0,
        ulong nbw1,
        ulong nbw2,
        ulong nbd0,
        ulong nbd1,
        float scale,
        int gated) {
    const int i0 = get_global_id(0);
    const int it = get_global_id(1);
    if (i0 >= n_embd || it >= n_tokens) {
        return;
    }

    x += offx;
    w += offw;
    dst += offd;

    float sum = 0.0f;
    for (int ih = 0; ih < hc; ++ih) {
        const float xv = *(global const float *) (x + i0 * nbx0 + ih * nbx1 + it * nbx2);
        float wv;
        if (gated) {
            const float gv = *(global const float *) (w + i0 * nbw0 + ih * nbw1 + it * nbw2);
            wv = 1.0f / (1.0f + exp(-gv));
        } else {
            wv = *(global const float *) (w + ih * nbw0 + it * nbw1);
        }
        sum = fma(xv, wv, sum);
    }

    *(global float *) (dst + i0 * nbd0 + it * nbd1) = scale * sum;
}

// result[i, dst, t] = x[i, t] * post[dst, t] + sum_src residual[i, src, t] * comb[dst, src, t]
// has_comb == 0: result[i, dst, t] = x[i, t] * post[dst, t] + residual[i, dst, t]
// one work-item per (i, dst, t): gid0 = i, gid1 = dst + hc*t
kernel void kernel_dsv4_hc_post_f32(
        global const char * x,
        ulong offx,
        global const char * res,
        ulong offr,
        global const char * post,
        ulong offp,
        global const char * comb,
        ulong offc,
        global char * dst,
        ulong offd,
        int n_embd,
        int hc,
        int n_tokens,
        ulong nbx0,
        ulong nbx1,
        ulong nbr0,
        ulong nbr1,
        ulong nbr2,
        ulong nbp0,
        ulong nbp1,
        ulong nbc0,
        ulong nbc1,
        ulong nbc2,
        ulong nbd0,
        ulong nbd1,
        ulong nbd2,
        int has_comb) {
    const int i0 = get_global_id(0);
    const int row = get_global_id(1);
    if (i0 >= n_embd || row >= hc * n_tokens) {
        return;
    }
    const int idst = row % hc;
    const int it = row / hc;

    x += offx;
    res += offr;
    post += offp;
    comb += offc;
    dst += offd;

    const float xv = *(global const float *) (x + i0 * nbx0 + it * nbx1);
    const float pv = *(global const float *) (post + idst * nbp0 + it * nbp1);

    float sum = xv * pv;
    if (has_comb) {
        for (int isrc = 0; isrc < hc; ++isrc) {
            const float rv = *(global const float *) (res + i0 * nbr0 + isrc * nbr1 + it * nbr2);
            const float cv = *(global const float *) (comb + idst * nbc0 + isrc * nbc1 + it * nbc2);
            sum = fma(rv, cv, sum);
        }
    } else {
        sum = sum + *(global const float *) (res + i0 * nbr0 + idst * nbr1 + it * nbr2);
    }

    *(global float *) (dst + i0 * nbd0 + idst * nbd1 + it * nbd2) = sum;
}

// comb[dst, src, t] from mixes[(2 + hc)*hc, t] (hc = 4): softmax over dst, + eps, then
// column / row normalizations (Sinkhorn), n_iter rounds. One work-item per token.
kernel void kernel_dsv4_hc_comb_f32(
        global const char * mixes,
        ulong offm,
        global const char * scale,
        ulong offs,
        global const char * base,
        ulong offb,
        global char * dst,
        ulong offd,
        int n_tokens,
        ulong nbm0,
        ulong nbm1,
        ulong nbs0,
        ulong nbb0,
        ulong nbd0,
        ulong nbd1,
        ulong nbd2,
        float eps,
        int n_iter) {
    const int it = get_global_id(0);
    if (it >= n_tokens) {
        return;
    }

    mixes += offm;
    scale += offs;
    base += offb;
    dst += offd;

    const int comb_offset = 2 * DSV4_HC;
    const float scale_comb = *(global const float *) (scale + 2 * nbs0);

    float comb[DSV4_HC * DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float mx = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            const float m = *(global const float *) (mixes + (comb_offset + idx) * nbm0 + it * nbm1);
            const float b = *(global const float *) (base + (comb_offset + idx) * nbb0);
            const float v = fma(m, scale_comb, b);
            comb[idx] = v;
            mx = fmax(mx, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            const float v = exp(comb[idx] - mx);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC * isrc;
            comb[idx] = fma(comb[idx], inv_sum, eps);
        }
    }

    for (int iter = 0; iter < n_iter; ++iter) {
        if (iter > 0) {
            // rows: normalize over dst for each src
            for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                float sum = eps;
                for (int idst = 0; idst < DSV4_HC; ++idst) {
                    sum += comb[idst + DSV4_HC * isrc];
                }
                const float inv_sum = 1.0f / sum;
                for (int idst = 0; idst < DSV4_HC; ++idst) {
                    comb[idst + DSV4_HC * isrc] *= inv_sum;
                }
            }
        }
        // cols: normalize over src for each dst
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            float sum = eps;
            for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                sum += comb[idst + DSV4_HC * isrc];
            }
            const float inv_sum = 1.0f / sum;
            for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
                comb[idst + DSV4_HC * isrc] *= inv_sum;
            }
        }
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            *(global float *) (dst + idst * nbd0 + isrc * nbd1 + it * nbd2) = comb[idst + DSV4_HC * isrc];
        }
    }
}
