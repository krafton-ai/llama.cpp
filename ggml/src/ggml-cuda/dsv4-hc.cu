#include "common.cuh"
#include "dsv4-hc.cuh"

#include <cfloat>


static constexpr int DSV4_HC = 4;

static __global__ void raon_hc_rms_tail_f32(
        const float * mean, const float * selector, float * dst, float eps, int64_t n_values) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) {
        return;
    }
    ggml_cuda_pdl_sync();
    const float rms = sqrtf(mean[i / 17] + eps);
    dst[i] = selector[i] / rms;
}

void ggml_cuda_op_raon_hc_rms_tail(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mean, const ggml_tensor * selector, ggml_tensor * dst, float eps) {
    const int64_t n_values = ggml_nelements(dst);
    const int block_size = 256;
    const dim3 grid((n_values + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(raon_hc_rms_tail_f32, params,
            (const float *) mean->data, (const float *) selector->data, (float *) dst->data, eps, n_values);
}

static __global__ void raon_hc_helmert_f32(
        const float * h, const float * z, float * dst, int64_t n_tokens) {
    ggml_cuda_pdl_lc();
    const int64_t token = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= n_tokens) {
        return;
    }
    ggml_cuda_pdl_sync();

    float tmp[12];
    for (int col = 0; col < 3; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum = fmaf(h[row*3 + k], z[token*9 + col*3 + k], sum);
            }
            tmp[col*4 + row] = sum;
        }
    }
    for (int col = 0; col < 4; ++col) {
        for (int row = 0; row < 4; ++row) {
            float sum = 0.0f;
            for (int k = 0; k < 3; ++k) {
                sum = fmaf(h[row*3 + k], tmp[k*4 + col], sum);
            }
            dst[token*16 + col*4 + row] = sum;
        }
    }
}

void ggml_cuda_op_raon_hc_helmert(ggml_backend_cuda_context & ctx,
        const ggml_tensor * h, const ggml_tensor * z, ggml_tensor * dst) {
    const int64_t n_tokens = z->ne[2];
    const int block_size = 128;
    const dim3 grid((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(raon_hc_helmert_f32, params,
            (const float *) h->data, (const float *) z->data, (float *) dst->data, n_tokens);
}

// kind: 0 = pre sigmoid, 1 = post sigmoid + magnitude, 2 = Helmert coordinates.
template <int kind>
static __global__ void raon_hc_affine_f32(
        const float * mixes, const float * scale, const float * base, float * dst,
        int64_t n_values, int64_t mixes_stride, float magnitude, float bias) {
    constexpr int width = kind == 2 ? 9 : DSV4_HC;
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) {
        return;
    }
    ggml_cuda_pdl_sync();

    const int64_t lane = i % width;
    const int64_t token = i / width;
    // The unfused MUL rounds before ADD, including cancellation-heavy inputs.
    volatile float product = mixes[token*mixes_stride + lane] * scale[0];
    float value = product + base[lane];
    if constexpr (kind != 2) {
        value = 1.0f / (1.0f + expf(-value));
    }
    if constexpr (kind == 1) {
        value = magnitude * value + bias;
    }
    dst[i] = value;
}

void ggml_cuda_op_raon_hc_affine(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mixes, const ggml_tensor * scale, const ggml_tensor * base,
        ggml_tensor * dst, int kind) {
    GGML_ASSERT(kind >= 0 && kind <= 2);
    const int64_t n_values = ggml_nelements(dst);
    const int block_size = 256;
    const dim3 grid((n_values + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    auto kernel = kind == 0 ? raon_hc_affine_f32<0>
                : kind == 1 ? raon_hc_affine_f32<1> : raon_hc_affine_f32<2>;
    const float magnitude = kind == 1 ? ggml_get_op_params_f32(dst, 0) : 1.0f;
    const float bias      = kind == 1 ? ggml_get_op_params_f32(dst, 1) : 0.0f;
    ggml_cuda_kernel_launch(kernel, params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data,
            (float *) dst->data, n_values, mixes->nb[1] / sizeof(float), magnitude, bias);
}

static __global__ void raon_hc_comb_f32(
        const float * e, const float * scale, const float * chart,
        const float * identity, float * dst, int64_t n_tokens) {
    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (it >= n_tokens) {
        return;
    }
    ggml_cuda_pdl_sync();

    float values[DSV4_HC*DSV4_HC];
    float maximum = -FLT_MAX;
    for (int j = 0; j < DSV4_HC*DSV4_HC; ++j) {
        values[j] = e[it*16 + j];
        const float ratio = values[j] / chart[j];
        maximum = max(maximum, -ratio);
    }
    const float rho = fminf(fmaxf(maximum, 0.0f), INFINITY) + 1.0f;
    const float gamma = 1.0f / (1.0f + expf(-scale[3]));
    for (int j = 0; j < DSV4_HC*DSV4_HC; ++j) {
        const float normalized = values[j] / rho;
        const float phi = normalized + chart[j];
        const float shifted = phi - identity[j];
        // Keep the separate MUL and ADD rounding of the unfused graph.
        volatile float weighted = shifted * gamma;
        dst[it*16 + j] = weighted + identity[j];
    }
}

void ggml_cuda_op_raon_hc_comb(ggml_backend_cuda_context & ctx,
        const ggml_tensor * e, const ggml_tensor * scale, const ggml_tensor * chart,
        const ggml_tensor * identity, ggml_tensor * dst) {
    const int64_t n_tokens = e->ne[2];
    const int block_size = 256;
    const dim3 grid((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(raon_hc_comb_f32, params,
            (const float *) e->data, (const float *) scale->data,
            (const float *) chart->data, (const float *) identity->data,
            (float *) dst->data, n_tokens);
}


static __device__ void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    for (int idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __device__ void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static __global__ void dsv4_hc_comb_f32(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter) {
    constexpr int comb_offset = 2*DSV4_HC;

    ggml_cuda_pdl_lc();
    const int64_t it = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;

    if (it >= n_tokens) {
        return;
    }

    ggml_cuda_pdl_sync();

    const float scale_comb = scale[2*ss0];
    float comb[DSV4_HC*DSV4_HC];

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        float max = -INFINITY;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
            comb[idx] = v;
            max = fmaxf(max, v);
        }

        float sum = 0.0f;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            const float v = expf(comb[idx] - max);
            comb[idx] = v;
            sum += v;
        }

        const float inv_sum = 1.0f / sum;
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            comb[idx] = comb[idx] * inv_sum + eps;
        }
    }

    dsv4_hc_comb_norm_cols(comb, eps);
    for (int32_t i = 1; i < n_iter; ++i) {
        dsv4_hc_comb_norm_rows(comb, eps);
        dsv4_hc_comb_norm_cols(comb, eps);
    }

    for (int isrc = 0; isrc < DSV4_HC; ++isrc) {
        for (int idst = 0; idst < DSV4_HC; ++idst) {
            const int idx = idst + DSV4_HC*isrc;
            dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
        }
    }
}

template <bool gated>
static __global__ void dsv4_hc_pre_f32(
        const float * x,
        const float * weights,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sx2,
        int64_t sw0,
        int64_t sw1,
        int64_t sw2,
        int64_t sd0,
        int64_t sd1,
        float   scale) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0 = ir % n_embd;
    const int64_t it = ir / n_embd;

    float sum = 0.0f;
    for (int64_t ih = 0; ih < hc; ++ih) {
        const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
        float wv;
        if constexpr (gated) {
            wv = 1.0f / (1.0f + expf(-weights[i0*sw0 + ih*sw1 + it*sw2]));
        } else {
            wv = weights[ih*sw0 + it*sw1];
        }
        sum += xv * wv;
    }

    dst[i0*sd0 + it*sd1] = scale * sum;
}

// gated: post = gate_scale_out*sigmoid(gate_scale_in*post), the fused SCALE -> SIGMOID -> SCALE chain of qwen4exp
template <bool has_comb, bool gated>
static __global__ void dsv4_hc_post_f32(
        const float * x,
        const float * residual,
        const float * post,
        const float * comb,
        float * dst,
        int64_t n_embd,
        int64_t hc,
        int64_t n_tokens,
        int64_t sx0,
        int64_t sx1,
        int64_t sr0,
        int64_t sr1,
        int64_t sr2,
        int64_t sp0,
        int64_t sp1,
        int64_t sc0,
        int64_t sc1,
        int64_t sc2,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float gate_scale_in,
        float gate_scale_out) {
    ggml_cuda_pdl_lc();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t nr = n_embd * hc * n_tokens;

    if (ir >= nr) {
        return;
    }

    ggml_cuda_pdl_sync();

    const int64_t i0   = ir % n_embd;
    const int64_t idst = (ir / n_embd) % hc;
    const int64_t it   = ir / (n_embd * hc);

    float p = post[idst*sp0 + it*sp1];
    if constexpr (gated) {
        p = gate_scale_out * (1.0f / (1.0f + expf(-(gate_scale_in * p))));
    }

    float sum = x[i0*sx0 + it*sx1] * p;
    if constexpr (has_comb) {
        for (int64_t isrc = 0; isrc < hc; ++isrc) {
            sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
        }
    } else {
        sum += residual[i0*sr0 + idst*sr1 + it*sr2];
    }

    dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
}

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    const int block_size = 256;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((n_tokens + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    ggml_cuda_kernel_launch(dsv4_hc_comb_f32, launch_params,
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / sizeof(float), nbm1 / sizeof(float),
            nbs0 / sizeof(float),
            nbb0 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            eps, n_iter);
}

void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    const int block_size = 256;
    const int64_t nr = n_embd * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = gated ? dsv4_hc_pre_f32<true> : dsv4_hc_pre_f32<false>;
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) weights->data, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float), nbx2 / sizeof(float),
            nbw0 / sizeof(float), nbw1 / sizeof(float), nbw2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float),
            scale);
}

void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in, const ggml_tensor * gate_scale_out) {
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * comb     = dst->src[3];

    // when fused, read the raw post weights from before the SCALE -> SIGMOID -> SCALE chain
    const bool          gated = gate_scale_in != nullptr;
    const ggml_tensor * post  = gated ? gate_scale_in->src[0] : dst->src[2];

    const float gate_in  = gated ? ggml_get_op_params_f32(gate_scale_in,  0) : 1.0f;
    const float gate_out = gated ? ggml_get_op_params_f32(gate_scale_out, 0) : 1.0f;

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(comb == nullptr || comb->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    const size_t nbc0 = comb ? comb->nb[0] : 0;
    const size_t nbc1 = comb ? comb->nb[1] : 0;
    const size_t nbc2 = comb ? comb->nb[2] : 0;

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    const int block_size = 256;
    const int64_t nr = n_embd * hc * n_tokens;
    const dim3 block_dims(block_size, 1, 1);
    const dim3 grid_dims((nr + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, ctx.stream());

    auto kernel = comb ? (gated ? dsv4_hc_post_f32<true,  true> : dsv4_hc_post_f32<true,  false>)
                       : (gated ? dsv4_hc_post_f32<false, true> : dsv4_hc_post_f32<false, false>);
    ggml_cuda_kernel_launch(kernel, launch_params,
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / sizeof(float), nbx1 / sizeof(float),
            nbr0 / sizeof(float), nbr1 / sizeof(float), nbr2 / sizeof(float),
            nbp0 / sizeof(float), nbp1 / sizeof(float),
            nbc0 / sizeof(float), nbc1 / sizeof(float), nbc2 / sizeof(float),
            nbd0 / sizeof(float), nbd1 / sizeof(float), nbd2 / sizeof(float),
            gate_in, gate_out);
}
