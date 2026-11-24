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

