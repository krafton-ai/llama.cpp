#include "raon-hc-fuse.cuh"
#include "ggml-impl.h"

#include <cstdlib>
#include <cstdio>
#include <vector>
#include <cfloat>

// Every kernel below must reproduce, bit for bit, what the separate ggml-cuda kernels compute.
// These sources are compiled with the same flags (-use_fast_math, fmad on) as the originals, so plain
// expressions are copied from the original kernels; products that were rounded in memory between two
// kernels use __fmul_rn/__fadd_rn so the compiler cannot contract them into an FMA.

static __device__ __forceinline__ float rhc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }  // unary.cu op_sigmoid

// A: SQR -> MEAN -> SCALE(scale, bias) -> SQRT -> DIV(mixraw, rms) -> MUL(scale_w) -> ADD(base) -> SIGMOID.
// outputs: mixes[17] (the DIV result, consumed later by the post/comb chains) and pre[4].
static __global__ void raon_hc_a_f32(const float * x, const int ncols, const float * mixraw, const int nmix,
        const float * scale_w, const float * base_w, const float sc, const float bi, const int npre,
        float * mixes, float * pre) {
    const int col = threadIdx.x;
    x += (size_t) blockIdx.x * ncols; mixraw += (size_t) blockIdx.x * nmix; mixes += (size_t) blockIdx.x * nmix; pre += (size_t) blockIdx.x * npre;  // one block per token

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const float v = x[i];
                temp[j] = __fmul_rn(v, v);      // SQR kernel output (rounded)
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] = __fadd_rn(sum_temp[j], temp[j]);
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum = __fadd_rn(sum, sum_temp[j]);
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    __shared__ float s_rms;
    __shared__ float s_mix[32];
    if (col == 0) {
        const float mean = sum / ncols;           // reduce_rows_f32<norm=true>
        const float t    = sc * mean + bi;        // scale_f32 (contracted like the original)
        s_rms = sqrtf(t);                         // op_sqrt
    }
    __syncthreads();
    if (col < nmix) {
        const float m = mixraw[col] / s_rms;      // op_div
        mixes[col] = m;
        s_mix[col] = m;
    }
    __syncthreads();
    if (col < npre) {
        pre[col] = rhc_sigmoid(__fadd_rn(__fmul_rn(s_mix[col], scale_w[0]), base_w[col]));
    }
}

// AP: A followed by DSV4_HC_PRE (n_tokens == 1): dst[i0] = scale * sum_ih x[i0 + ih*n_embd] * pre[ih]  (same statements as dsv4_hc_pre_f32<false>)
static __global__ void raon_hc_ap_f32(const float * x, const int ncols, const float * mixraw, const int nmix,
        const float * scale_w, const float * base_w, const float sc, const float bi, const int npre,
        float * mixes, float * pre,
        const float * px, float * pdst, const int64_t n_embd, const int64_t hc, const int64_t sx0, const int64_t sx1, const int64_t sw0, const float pscale) {
    const int col = threadIdx.x;
    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };

    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                const float v = x[i];
                temp[j] = __fmul_rn(v, v);
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] = __fadd_rn(sum_temp[j], temp[j]);
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum = __fadd_rn(sum, sum_temp[j]);
    }

    __shared__ float shared_vals[32];
    sum = block_reduce<block_reduce_method::SUM>(sum, shared_vals);

    __shared__ float s_rms;
    __shared__ float s_mix[32];
    __shared__ float s_pre[32];
    if (col == 0) {
        const float mean = sum / ncols;
        const float t    = sc * mean + bi;
        s_rms = sqrtf(t);
    }
    __syncthreads();
    if (col < nmix) {
        const float m = mixraw[col] / s_rms;
        mixes[col] = m;
        s_mix[col] = m;
    }
    __syncthreads();
    if (col < npre) {
        const float pv = rhc_sigmoid(__fadd_rn(__fmul_rn(s_mix[col], scale_w[0]), base_w[col]));
        pre[col] = pv;
        s_pre[col] = pv;
    }
    __syncthreads();
    for (int64_t i0 = col; i0 < n_embd; i0 += blockDim.x) {
        float s = 0.0f;
        for (int64_t ih = 0; ih < hc; ++ih) {
            const float xv = px[i0 * sx0 + ih * sx1];
            const float wv = s_pre[ih * sw0];
            s += xv * wv;
        }
        pdst[i0] = pscale * s;
    }
}

// B: MUL(z, scale) -> ADD(base) -> SIGMOID -> SCALE(s, b)   (hc_post weights)
static __global__ void raon_hc_b_f32(const float * z, const float * scale_w, const float * base_w, const float sc, const float bi,
        const int n, const int zstride, float * dst) {
    const int i = threadIdx.x;
    z += (size_t) blockIdx.x * zstride; dst += (size_t) blockIdx.x * n;
    if (i < n) {
        const float s = rhc_sigmoid(__fadd_rn(__fmul_rn(z[i], scale_w[0]), base_w[i]));
        dst[i] = sc * s + bi;
    }
}

// Z: MUL(z, scale) -> ADD(base) -> CONT
static __global__ void raon_hc_z_f32(const float * z, const float * scale_w, const float * base_w, const int n, const int zstride, float * dst) {
    const int i = threadIdx.x;
    z += (size_t) blockIdx.x * zstride; dst += (size_t) blockIdx.x * n;
    if (i < n) {
        dst[i] = __fadd_rn(__fmul_rn(z[i], scale_w[0]), base_w[i]);
    }
}

// C: DIV(e, chart) -> NEG -> POOL_1D(max, 16) -> CLAMP(0, inf) -> SCALE(1, 1) -> DIV(e, rho) -> ADD(chart) -> SUB(identity)
//    -> MUL(sigmoid(gamma)) -> ADD(identity)         (all 4x4, one token)
static __global__ void raon_hc_c_f32(const float * e, const float * chart, const float * ident, const float * gamma_w,
        const float cl_min, const float cl_max, const float sc, const float bi, float * dst) {
    __shared__ float s_neg[16];
    const int k = threadIdx.x;   // 16 threads
    e += (size_t) blockIdx.x * 16; dst += (size_t) blockIdx.x * 16;   // one block per token; chart/ident/gamma are shared
    const float t = e[k] / chart[k];
    s_neg[k] = -t;
    __syncthreads();
    float rho = -FLT_MAX;                       // pool1d_nchw_kernel, POOL_MAX
    for (int i = 0; i < 16; ++i) {
        rho = max(rho, s_neg[i]);
    }
    rho = fminf(fmaxf(rho, cl_min), cl_max);    // op_clamp
    rho = sc * rho + bi;                        // scale_f32 (contracted)
    const float phi   = __fadd_rn(e[k] / rho, chart[k]);
    const float diff  = __fsub_rn(phi, ident[k]);
    const float gamma = rhc_sigmoid(gamma_w[0]);
    dst[k] = __fadd_rn(__fmul_rn(diff, gamma), ident[k]);
}



// mHC combination for one token.
// D: MUL(mix9, scale) -> ADD(base) -> CONT[3,3] -> MUL_MAT(C[3,4], z) [4,3] -> TRANSPOSE+CONT [3,4] -> MUL_MAT(C, .) [4,4] -> hc_c chain (DIV ... ADD).
// The ggml cuBLAS sgemm with K = 3 (ampere_sgemm_32x32_sliced1x4_tn) is reproduced bitwise by a zero-initialised ascending FMA chain: fma(a2,b2,fma(a1,b1,fma(a0,b0,0))) (checked on the real tensors; a standalone cublasSgemm test picks a different kernel and is not representative).
static __device__ __forceinline__ float rhc_dot3(const float a0, const float a1, const float a2, const float b0, const float b1, const float b2, const int var) {
    if (var == 0) { return __fmaf_rn(a2, b2, __fmaf_rn(a1, b1, __fmaf_rn(a0, b0, 0.0f))); }                        // zero-initialised ascending fma chain (n == 1 cublas kernel)
    return __fadd_rn(__fadd_rn(__fadd_rn(0.0f, __fmul_rn(a0, b0)), __fmul_rn(a1, b1)), __fmul_rn(a2, b2));             // separately rounded products (batched cublas kernel)
}
static __global__ void raon_hc_d_f32(const float * mix, const int mix_stride, const float * scale_w, const float * base_w, const float * cm, const float * chart, const float * ident,
        const float * gamma_w, const float cl_min, const float cl_max, const float sc, const float bi, float * dst, const int var) {
    __shared__ float s_z[9], s_r[12], s_rt[12], s_e[16], s_neg[16];
    const int k = threadIdx.x;   // 16 threads, one token per block
    mix += (size_t) blockIdx.x * mix_stride; dst += (size_t) blockIdx.x * 16;
    if (k < 9) {
        s_z[k] = __fadd_rn(__fmul_rn(mix[k], scale_w[0]), base_w[k]);
    }
    __syncthreads();
    if (k < 12) {                // sgemm 1: r[m + 4n] = sum_j cm[j + 3m] * z[j + 3n]
        const int m = k % 4, n = k / 4;
        s_r[k] = rhc_dot3(cm[0 + m * 3], cm[1 + m * 3], cm[2 + m * 3], s_z[0 + n * 3], s_z[1 + n * 3], s_z[2 + n * 3], var);
    }
    __syncthreads();
    if (k < 12) {                // transpose: rt[i0 + 3*i1] = r[i1 + 4*i0]
        const int i0 = k % 3, i1 = k / 3;
        s_rt[k] = s_r[i1 + i0 * 4];
    }
    __syncthreads();
    {                            // sgemm 2: e[m + 4n] = sum_j cm[j + 3m] * rt[j + 3n]
        const int m = k % 4, n = k / 4;
        s_e[k] = rhc_dot3(cm[0 + m * 3], cm[1 + m * 3], cm[2 + m * 3], s_rt[0 + n * 3], s_rt[1 + n * 3], s_rt[2 + n * 3], var);
    }
    __syncthreads();
    const float t = s_e[k] / chart[k];
    s_neg[k] = -t;
    __syncthreads();
    float rho = -FLT_MAX;
    for (int i = 0; i < 16; ++i) {
        rho = max(rho, s_neg[i]);
    }
    rho = fminf(fmaxf(rho, cl_min), cl_max);
    rho = sc * rho + bi;
    const float phi   = __fadd_rn(s_e[k] / rho, chart[k]);
    const float diff  = __fsub_rn(phi, ident[k]);
    const float gamma = rhc_sigmoid(gamma_w[0]);
    dst[k] = __fadd_rn(__fmul_rn(diff, gamma), ident[k]);
}

// mHC combination and post update for one token.
// DP: the D chain above (every block recomputes the 4x4 comb matrix in shared memory: 16 active threads, uniform barriers) followed by
// dsv4_hc_post_f32<true,false>: dst[i0 + idst*n_embd] = x[i0] * post[idst] + sum_isrc residual[i0, isrc] * comb[idst, isrc]  (same statements as the original kernel).
static __global__ void raon_hc_dp_f32(const float * mix, const float * scale_w, const float * base_w, const float * cm, const float * chart, const float * ident,
        const float * gamma_w, const float cl_min, const float cl_max, const float sc, const float bi, const int var,
        const float * x, const float * residual, const float * post, float * dst,
        const int64_t n_embd, const int64_t hc, const int64_t sx0, const int64_t sr0, const int64_t sr1, const int64_t sp0,
        const int64_t sc0, const int64_t sc1, const int64_t sd0, const int64_t sd1) {
    __shared__ float s_z[9], s_r[12], s_rt[12], s_e[16], s_neg[16], s_comb[16];
    const int k = threadIdx.x;
    const bool act = k < 16;
    if (k < 9) {
        s_z[k] = __fadd_rn(__fmul_rn(mix[k], scale_w[0]), base_w[k]);
    }
    __syncthreads();
    if (k < 12) {
        const int m = k % 4, n = k / 4;
        s_r[k] = rhc_dot3(cm[0 + m * 3], cm[1 + m * 3], cm[2 + m * 3], s_z[0 + n * 3], s_z[1 + n * 3], s_z[2 + n * 3], var);
    }
    __syncthreads();
    if (k < 12) {
        const int i0 = k % 3, i1 = k / 3;
        s_rt[k] = s_r[i1 + i0 * 4];
    }
    __syncthreads();
    if (act) {
        const int m = k % 4, n = k / 4;
        s_e[k] = rhc_dot3(cm[0 + m * 3], cm[1 + m * 3], cm[2 + m * 3], s_rt[0 + n * 3], s_rt[1 + n * 3], s_rt[2 + n * 3], var);
    }
    __syncthreads();
    if (act) {
        const float t = s_e[k] / chart[k];
        s_neg[k] = -t;
    }
    __syncthreads();
    if (act) {
        float rho = -FLT_MAX;
        for (int i = 0; i < 16; ++i) {
            rho = max(rho, s_neg[i]);
        }
        rho = fminf(fmaxf(rho, cl_min), cl_max);
        rho = sc * rho + bi;
        const float phi   = __fadd_rn(s_e[k] / rho, chart[k]);
        const float diff  = __fsub_rn(phi, ident[k]);
        const float gamma = rhc_sigmoid(gamma_w[0]);
        s_comb[k] = __fadd_rn(__fmul_rn(diff, gamma), ident[k]);
    }
    __syncthreads();
    const int64_t ir = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (ir >= n_embd * hc) {
        return;
    }
    const int64_t i0   = ir % n_embd;
    const int64_t idst = ir / n_embd;
    const float p = post[idst * sp0];
    float sum = x[i0 * sx0] * p;
    for (int64_t isrc = 0; isrc < hc; ++isrc) {
        sum += residual[i0 * sr0 + isrc * sr1] * s_comb[idst * sc0 + isrc * sc1];
    }
    dst[i0 * sd0 + idst * sd1] = sum;
}

static constexpr int raon_fusion_max_tokens = 100000;

// the fused kernels read inputs (many of them broadcast per head/token) and write the output in one launch. The ggml allocator may place the
// output over the buffer of an input whose consumer disappeared in the fusion; the original kernel chain was safe (one launch per step), the fused one races.
// Decline the fusion unless the output does not overlap the input at all (or, for same-index element-wise inputs, is exactly the same buffer).
static bool rhc_ov(const ggml_tensor * a, const ggml_tensor * b) {
    if (!a || !b || !a->data || !b->data) { return false; }
    const char * a0 = (const char *) a->data, * a1 = a0 + ggml_nbytes(a), * b0 = (const char *) b->data, * b1 = b0 + ggml_nbytes(b);
    return a0 < b1 && b0 < a1;
}
static bool rhc_ok(const ggml_tensor * out, const ggml_tensor * in, bool exact_ok) {
    if (!rhc_ov(out, in)) { return true; }
    return exact_ok && out->data == in->data && ggml_nbytes(out) == ggml_nbytes(in);
}

static bool rhc_f32c(const ggml_tensor * t) { return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t); }

bool ggml_cuda_raon_hc_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * n_skip) {
    if (i + 14 > cgraph->n_nodes) {
        return false;
    }
    ggml_tensor ** nd = cgraph->nodes + i;
    cudaStream_t stream = ctx.stream();

    if (nd[0]->op == GGML_OP_MUL && i + 22 <= cgraph->n_nodes) {
        static const enum ggml_op ops[22] = { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT,
            GGML_OP_DIV, GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_POOL_1D, GGML_OP_CLAMP, GGML_OP_SCALE, GGML_OP_RESHAPE, GGML_OP_DIV, GGML_OP_ADD, GGML_OP_SUB,
            GGML_OP_VIEW, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
        const int outputs[1] = { i + 21 };
        ggml_tensor * mul = nd[0], * add = nd[2], * ct = nd[3], * m1 = nd[4], * ct2 = nd[6], * m2 = nd[7], * d1 = nd[8], * pool = nd[11], * cl = nd[12], * scn = nd[13],
                    * d2 = nd[15], * a1 = nd[16], * sb = nd[17], * sg = nd[19], * ml = nd[20], * a2 = nd[21];
        const int64_t nt = mul->src[0]->ne[1];
        bool ok = true;
        for (int q = 0; q < 22 && ok; ++q) { ok = nd[q]->op == ops[q]; }
        if (ok && nd[9]->op == GGML_OP_UNARY && ggml_get_unary_op(nd[9]) == GGML_UNARY_OP_NEG && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID &&
                ggml_can_fuse_subgraph(cgraph, i, 22, ops, outputs, 1) &&
                mul->src[0]->type == GGML_TYPE_F32 && mul->src[0]->ne[0] == 9 && mul->src[0]->ne[1] >= 1 && mul->src[0]->ne[1] <= raon_fusion_max_tokens && mul->src[0]->ne[2] == 1 && mul->src[0]->ne[3] == 1 && mul->src[0]->nb[0] == 4 && mul->src[0]->nb[1] % 4 == 0 && ggml_nelements(mul->src[1]) == 1 &&
                add->src[0] == mul && add->src[1]->type == GGML_TYPE_F32 && ggml_nelements(add->src[1]) == 9 && rhc_f32c(add->src[1]) &&
                ct->src[0] == add && rhc_f32c(ct) && ct->ne[0] == 3 && ct->ne[1] == 3 && ct->ne[2] == nt && ct->ne[3] == 1 &&
                m1->src[1] == ct && m1->src[0]->type == GGML_TYPE_F32 && rhc_f32c(m1->src[0]) && m1->src[0]->ne[0] == 3 && m1->src[0]->ne[1] == 4 && m1->src[0]->ne[2] == 1 && m1->src[0]->ne[3] == 1 &&
                m1->ne[0] == 4 && m1->ne[1] == 3 && m1->ne[2] == nt && m1->ne[3] == 1 && rhc_f32c(m1) &&
                nd[5]->src[0] == m1 && ct2->src[0] == nd[5] && rhc_f32c(ct2) && ct2->ne[0] == 3 && ct2->ne[1] == 4 && ct2->ne[2] == nt && ct2->ne[3] == 1 &&
                m2->src[1] == ct2 && m2->src[0] == m1->src[0] && m2->ne[0] == 4 && m2->ne[1] == 4 && m2->ne[2] == nt && m2->ne[3] == 1 && rhc_f32c(m2) &&
                d1->src[0] == m2 && d2->src[0] == m2 && d1->src[1] == a1->src[1] && sb->src[1] == a2->src[1] &&
                rhc_f32c(d1->src[1]) && rhc_f32c(sb->src[1]) && ggml_nelements(d1->src[1]) == 16 && ggml_nelements(sb->src[1]) == 16 && rhc_f32c(a2) && a2->ne[0] == 4 && a2->ne[1] == 4 && a2->ne[2] == nt &&
                ggml_nelements(sg->src[0]) == 1 && sg->src[0]->type == GGML_TYPE_F32 && ml->src[0] == sb && ml->src[1] == sg &&
                pool->op_params != nullptr && ((const int32_t *) pool->op_params)[0] == GGML_OP_POOL_MAX &&
                ((const int32_t *) pool->op_params)[1] == 16 && ((const int32_t *) pool->op_params)[2] == 16 && ((const int32_t *) pool->op_params)[3] == 0 && rhc_ok(a2, mul->src[0], false) && rhc_ok(a2, mul->src[1], false) && rhc_ok(a2, add->src[1], false) && rhc_ok(a2, m1->src[0], false) && rhc_ok(a2, d1->src[1], false) && rhc_ok(a2, sb->src[1], false) && rhc_ok(a2, sg->src[0], false)) {
            float mn, mx, s, b;
            memcpy(&mn, (const float *) cl->op_params + 0, sizeof(float));
            memcpy(&mx, (const float *) cl->op_params + 1, sizeof(float));
            memcpy(&s,  (const float *) scn->op_params + 0, sizeof(float));
            memcpy(&b,  (const float *) scn->op_params + 1, sizeof(float));
            // n == 1: the ggml cublas sgemm is a zero-initialised ascending fma chain; n > 1: the batched cublas kernel sums separately rounded
            // products in ascending order
            const int dot_var = nt == 1 ? 0 : 1;
            // n_tokens == 1 and the chain is directly consumed by the DSV4_HC_POST that follows -> one kernel for both (a2 is not written: it has no other consumer)
            if (nt == 1 && i + 23 <= cgraph->n_nodes && nd[22]->op == GGML_OP_DSV4_HC_POST) {
                static const enum ggml_op ops23[23] = { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_CONT, GGML_OP_MUL_MAT, GGML_OP_TRANSPOSE, GGML_OP_CONT, GGML_OP_MUL_MAT,
                    GGML_OP_DIV, GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_POOL_1D, GGML_OP_CLAMP, GGML_OP_SCALE, GGML_OP_RESHAPE, GGML_OP_DIV, GGML_OP_ADD, GGML_OP_SUB,
                    GGML_OP_VIEW, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_DSV4_HC_POST };
                const int outputs23[1] = { i + 22 };
                ggml_tensor * po = nd[22];
                const ggml_tensor * px = po->src[0], * pr = po->src[1], * pp = po->src[2];
                if (po->src[3] == a2 && px && pr && pp && px->type == GGML_TYPE_F32 && pr->type == GGML_TYPE_F32 && pp->type == GGML_TYPE_F32 && po->type == GGML_TYPE_F32 &&
                        px->ne[1] == 1 && pr->ne[1] == 4 && pr->ne[2] == 1 && px->ne[0] == pr->ne[0] && po->ne[0] == px->ne[0] && po->ne[1] == 4 && po->ne[2] == 1 &&
                        px->nb[0] % 4 == 0 && pr->nb[0] % 4 == 0 && pr->nb[1] % 4 == 0 && pp->nb[0] % 4 == 0 && po->nb[0] % 4 == 0 && po->nb[1] % 4 == 0 &&
                        a2->nb[0] == 4 && a2->nb[1] == 16 &&
                        ggml_can_fuse_subgraph(cgraph, i, 23, ops23, outputs23, 1) &&
                        rhc_ok(po, mul->src[0], false) && rhc_ok(po, mul->src[1], false) && rhc_ok(po, add->src[1], false) && rhc_ok(po, m1->src[0], false) &&
                        rhc_ok(po, d1->src[1], false) && rhc_ok(po, sb->src[1], false) && rhc_ok(po, sg->src[0], false) && rhc_ok(po, pp, false)) {
                    const int64_t nr = po->ne[0] * 4;
                    raon_hc_dp_f32<<<(int) ((nr + 255) / 256), 256, 0, stream>>>((const float *) mul->src[0]->data, (const float *) mul->src[1]->data, (const float *) add->src[1]->data,
                            (const float *) m1->src[0]->data, (const float *) d1->src[1]->data, (const float *) sb->src[1]->data, (const float *) sg->src[0]->data, mn, mx, s, b, dot_var,
                            (const float *) px->data, (const float *) pr->data, (const float *) pp->data, (float *) po->data,
                            po->ne[0], 4, (int64_t) (px->nb[0] / 4), (int64_t) (pr->nb[0] / 4), (int64_t) (pr->nb[1] / 4), (int64_t) (pp->nb[0] / 4),
                            (int64_t) (a2->nb[0] / 4), (int64_t) (a2->nb[1] / 4), (int64_t) (po->nb[0] / 4), (int64_t) (po->nb[1] / 4));
                    *n_skip = 22;
                    return true;
                }
            }
            raon_hc_d_f32<<<(int) nt, 16, 0, stream>>>((const float *) mul->src[0]->data, (int) (mul->src[0]->nb[1] / 4), (const float *) mul->src[1]->data, (const float *) add->src[1]->data,
                    (const float *) m1->src[0]->data, (const float *) d1->src[1]->data, (const float *) sb->src[1]->data, (const float *) sg->src[0]->data, mn, mx, s, b, (float *) a2->data, dot_var);
            *n_skip = 21;
            return true;
        }
    }

    if (nd[0]->op == GGML_OP_SQR) {
        static const enum ggml_op ops[11] = { GGML_OP_SQR, GGML_OP_MEAN, GGML_OP_SCALE, GGML_OP_SQRT, GGML_OP_DIV, GGML_OP_VIEW,
            GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_UNARY };
        const int outputs[2] = { i + 4, i + 10 };
        ggml_tensor * x = nd[0]->src[0], * dv = nd[4], * mul = nd[7], * add = nd[9], * sg = nd[10];
        if (nd[10]->op == GGML_OP_UNARY && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID &&
                ggml_can_fuse_subgraph(cgraph, i, 11, ops, outputs, 2) &&
                rhc_f32c(x) && x->ne[1] >= 1 && x->ne[1] <= raon_fusion_max_tokens && x->ne[2] == 1 && x->ne[3] == 1 &&
                dv->src[1] == nd[3] && nd[3]->src[0] == nd[2] && nd[2]->src[0] == nd[1] && nd[1]->src[0] == nd[0] &&
                dv->src[0]->type == GGML_TYPE_F32 && dv->src[0]->ne[0] <= 32 && dv->src[0]->ne[1] == x->ne[1] && rhc_f32c(dv->src[0]) && dv->ne[1] == x->ne[1] && sg->ne[1] == x->ne[1] && sg->ne[2] == 1 && sg->ne[3] == 1 && mul->src[0]->ne[0] == sg->ne[0] &&
                dv->ne[0] == dv->src[0]->ne[0] && rhc_f32c(dv) && rhc_f32c(sg) && sg->ne[0] <= 32 &&
                mul->src[1]->type == GGML_TYPE_F32 && add->src[1]->type == GGML_TYPE_F32 && ggml_nelements(mul->src[1]) == 1 &&
                ggml_nelements(add->src[1]) == sg->ne[0] && mul->src[0]->view_src == dv && ggml_nelements(mul->src[0]) == ggml_nelements(sg) &&
                mul->src[0]->view_offs == 0 && nd[3]->op_params != nullptr && rhc_ok(dv, dv->src[0], true) && rhc_ok(dv, x, false) && rhc_ok(dv, mul->src[1], false) && rhc_ok(dv, add->src[1], false) && rhc_ok(sg, x, false) && rhc_ok(sg, dv->src[0], false) && rhc_ok(sg, mul->src[1], false) && rhc_ok(sg, add->src[1], false)) {
            float sc, bi;
            memcpy(&sc, (const float *) nd[2]->op_params + 0, sizeof(float));
            memcpy(&bi, (const float *) nd[2]->op_params + 1, sizeof(float));
            // ggml mean (reduce_rows_f32): 512 threads when nrows/nsm < 2, else 128 (ncols >= 1024) / 32; the reduction tree must match
            const int nsm_ = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
            const int a_threads = ((int) x->ne[1] / nsm_) < 2 ? 512 : (x->ne[0] < 1024 ? 32 : 128);
            // n_tokens == 1 and the next node is the DSV4_HC_PRE that consumes the sigmoid output -> same kernel
            if (x->ne[1] == 1 && sg->ne[0] == 4 && i + 12 <= cgraph->n_nodes && nd[11]->op == GGML_OP_DSV4_HC_PRE) {
                static const enum ggml_op ops12[12] = { GGML_OP_SQR, GGML_OP_MEAN, GGML_OP_SCALE, GGML_OP_SQRT, GGML_OP_DIV, GGML_OP_VIEW,
                    GGML_OP_VIEW, GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_DSV4_HC_PRE };
                const int outputs12[3] = { i + 4, i + 10, i + 11 };
                ggml_tensor * pr = nd[11];
                const ggml_tensor * px = pr->src[0], * pw = pr->src[1];
                const bool gated = ggml_get_op_params_i32(pr, 1) != 0;
                if (!gated && px && pw && px->type == GGML_TYPE_F32 && pw->type == GGML_TYPE_F32 && pr->type == GGML_TYPE_F32 && px->data == x->data && px->ne[0] == 2560 && px->ne[1] == 4 &&
                        px->ne[2] == 1 && x->ne[0] == px->ne[0] * px->ne[1] && px->nb[0] % 4 == 0 && px->nb[1] % 4 == 0 && pw->data == sg->data && pw->nb[0] % 4 == 0 && pw->ne[0] == 4 && pw->ne[1] == 1 &&
                        pr->ne[0] == px->ne[0] && pr->ne[1] == 1 && pr->nb[0] == 4 && ggml_can_fuse_subgraph(cgraph, i, 12, ops12, outputs12, 3) &&
                        rhc_ok(pr, x, false) && rhc_ok(pr, dv->src[0], false) && rhc_ok(pr, mul->src[1], false) && rhc_ok(pr, add->src[1], false) && rhc_ok(pr, dv, false) && rhc_ok(pr, sg, false)) {
                    raon_hc_ap_f32<<<1, a_threads, 0, stream>>>((const float *) x->data, (int) x->ne[0], (const float *) dv->src[0]->data, (int) dv->ne[0],
                            (const float *) mul->src[1]->data, (const float *) add->src[1]->data, sc, bi, (int) sg->ne[0], (float *) dv->data, (float *) sg->data,
                            (const float *) px->data, (float *) pr->data, px->ne[0], 4, (int64_t) (px->nb[0] / 4), (int64_t) (px->nb[1] / 4), (int64_t) (pw->nb[0] / 4),
                            ggml_get_op_params_f32(pr, 0));
                    *n_skip = 11;
                    return true;
                }
            }
            raon_hc_a_f32<<<(int) x->ne[1], a_threads, 0, stream>>>((const float *) x->data, (int) x->ne[0], (const float *) dv->src[0]->data, (int) dv->ne[0],
                    (const float *) mul->src[1]->data, (const float *) add->src[1]->data, sc, bi, (int) sg->ne[0],
                    (float *) dv->data, (float *) sg->data);
            *n_skip = 10;
            return true;
        }
    }

    if (nd[0]->op == GGML_OP_MUL) {
        static const enum ggml_op ops[5] = { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_UNARY, GGML_OP_SCALE };
        const int outputs[1] = { i + 4 };
        ggml_tensor * mul = nd[0], * add = nd[2], * sg = nd[3], * sc = nd[4];
        if (nd[3]->op == GGML_OP_UNARY && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID && nd[4]->op == GGML_OP_SCALE &&
                ggml_can_fuse_subgraph(cgraph, i, 5, ops, outputs, 1) &&
                rhc_f32c(sc) && sc->ne[0] <= 32 && sc->ne[1] >= 1 && sc->ne[1] <= raon_fusion_max_tokens && sc->ne[2] == 1 && sc->ne[3] == 1 && mul->src[0]->type == GGML_TYPE_F32 && ggml_nelements(mul->src[0]) == ggml_nelements(sc) && mul->src[0]->ne[0] == sc->ne[0] && mul->src[0]->nb[0] == 4 && mul->src[0]->nb[1] % 4 == 0 &&
                ggml_nelements(mul->src[1]) == 1 && ggml_nelements(add->src[1]) == sc->ne[0] && add->src[1]->type == GGML_TYPE_F32 &&
                add->src[0] == mul && rhc_ok(sc, mul->src[0], true) && rhc_ok(sc, mul->src[1], false) && rhc_ok(sc, add->src[1], false)) {
            float s, b;
            memcpy(&s, (const float *) sc->op_params + 0, sizeof(float));
            memcpy(&b, (const float *) sc->op_params + 1, sizeof(float));
            raon_hc_b_f32<<<(int) sc->ne[1], 32, 0, stream>>>((const float *) mul->src[0]->data, (const float *) mul->src[1]->data,
                    (const float *) add->src[1]->data, s, b, (int) sc->ne[0], (int) (mul->src[0]->nb[1] / 4), (float *) sc->data);
            *n_skip = 4;
            return true;
        }
    }

    if (nd[0]->op == GGML_OP_MUL) {
        static const enum ggml_op ops[4] = { GGML_OP_MUL, GGML_OP_VIEW, GGML_OP_ADD, GGML_OP_CONT };
        const int outputs[1] = { i + 3 };
        ggml_tensor * mul = nd[0], * add = nd[2], * ct = nd[3];
        if (ggml_can_fuse_subgraph(cgraph, i, 4, ops, outputs, 1) && rhc_f32c(ct) && ct->ne[0] <= 32 && ct->ne[1] >= 1 && ct->ne[1] <= raon_fusion_max_tokens && ct->ne[2] == 1 && ct->ne[3] == 1 &&
                mul->src[0]->type == GGML_TYPE_F32 && ggml_nelements(mul->src[0]) == ggml_nelements(ct) && mul->src[0]->ne[0] == ct->ne[0] && mul->src[0]->nb[0] == 4 && mul->src[0]->nb[1] % 4 == 0 &&
                ggml_nelements(mul->src[1]) == 1 && ggml_nelements(add->src[1]) == ct->ne[0] && add->src[1]->type == GGML_TYPE_F32 &&
                add->src[0] == mul && ct->src[0] == add && rhc_ok(ct, mul->src[0], true) && rhc_ok(ct, mul->src[1], false) && rhc_ok(ct, add->src[1], false)) {
            raon_hc_z_f32<<<(int) ct->ne[1], 32, 0, stream>>>((const float *) mul->src[0]->data, (const float *) mul->src[1]->data,
                    (const float *) add->src[1]->data, (int) ct->ne[0], (int) (mul->src[0]->nb[1] / 4), (float *) ct->data);
            *n_skip = 3;
            return true;
        }
    }

    if (nd[0]->op == GGML_OP_DIV && i + 14 <= cgraph->n_nodes) {
        static const enum ggml_op ops[14] = { GGML_OP_DIV, GGML_OP_UNARY, GGML_OP_RESHAPE, GGML_OP_POOL_1D, GGML_OP_CLAMP, GGML_OP_SCALE,
            GGML_OP_RESHAPE, GGML_OP_DIV, GGML_OP_ADD, GGML_OP_SUB, GGML_OP_VIEW, GGML_OP_UNARY, GGML_OP_MUL, GGML_OP_ADD };
        const int outputs[1] = { i + 13 };
        ggml_tensor * d1 = nd[0], * pool = nd[3], * cl = nd[4], * scn = nd[5], * d2 = nd[7], * a1 = nd[8], * sb = nd[9], * sg = nd[11], * ml = nd[12], * a2 = nd[13];
        if (nd[11]->op == GGML_OP_UNARY && ggml_get_unary_op(sg) == GGML_UNARY_OP_SIGMOID &&
                nd[1]->op == GGML_OP_UNARY && ggml_get_unary_op(nd[1]) == GGML_UNARY_OP_NEG &&
                ggml_can_fuse_subgraph(cgraph, i, 14, ops, outputs, 1) &&
                d1->ne[0] == 4 && d1->ne[1] == 4 && d1->ne[2] >= 1 && d1->ne[2] <= raon_fusion_max_tokens && d1->ne[3] == 1 && rhc_f32c(d1) && rhc_f32c(a2) && a2->ne[2] == d1->ne[2] && ggml_nelements(d1->src[1]) == 16 && ggml_nelements(sb->src[1]) == 16 &&
                d1->src[0] == d2->src[0] && d1->src[1] == a1->src[1] && sb->src[1] == a2->src[1] &&
                rhc_f32c(d1->src[0]) && rhc_f32c(d1->src[1]) && rhc_f32c(sb->src[1]) && ggml_nelements(sg->src[0]) == 1 &&
                sg->src[0]->type == GGML_TYPE_F32 && ml->src[0] == sb && ml->src[1] == sg &&
                pool->op_params != nullptr && ((const int32_t *) pool->op_params)[0] == GGML_OP_POOL_MAX &&
                ((const int32_t *) pool->op_params)[1] == 16 && ((const int32_t *) pool->op_params)[2] == 16 && ((const int32_t *) pool->op_params)[3] == 0 && rhc_ok(a2, d1->src[0], true) && rhc_ok(a2, d1->src[1], false) && rhc_ok(a2, sb->src[1], false) && rhc_ok(a2, sg->src[0], false)) {
            float mn, mx, s, b;
            memcpy(&mn, (const float *) cl->op_params + 0, sizeof(float));
            memcpy(&mx, (const float *) cl->op_params + 1, sizeof(float));
            memcpy(&s,  (const float *) scn->op_params + 0, sizeof(float));
            memcpy(&b,  (const float *) scn->op_params + 1, sizeof(float));
            raon_hc_c_f32<<<(int) d1->ne[2], 16, 0, stream>>>((const float *) d1->src[0]->data, (const float *) d1->src[1]->data,
                    (const float *) sb->src[1]->data, (const float *) sg->src[0]->data, mn, mx, s, b, (float *) a2->data);
            *n_skip = 13;
            return true;
        }
    }

    return false;
}

