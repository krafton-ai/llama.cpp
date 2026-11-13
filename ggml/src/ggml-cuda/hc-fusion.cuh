#pragma once

#include "common.cuh"

// Fused kernels for hyper-connection (mHC) subgraphs built from standard ops. The matchers are in ggml-cuda.cu;
// every function computes only the output nodes of its subgraph (the intermediate nodes are not written).

// RMS_NORM(x) -> MUL_MAT(w, .) -> ADD(., bias) [-> VIEW(cols 0..n_sig) -> SIGMOID]
// w has few rows: the dot products are split along the input (split-K) so that one token keeps many SMs busy
void ggml_cuda_op_rms_norm_mat_vec_add(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_mat, ggml_tensor * add, ggml_tensor * sig);

// DSV4_HC_PRE(x, pre) -> RMS_NORM -> MUL(w)
// dst_q8_1 (one token): also writes MUL quantized to q8_1 for mul_mat_vec_q, zero padded to ncols_padded
void ggml_cuda_op_hc_pre_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_tensor * hc_pre,
        const ggml_tensor * rms_norm, ggml_tensor * mul, void * dst_q8_1 = nullptr, int64_t ncols_padded = 0);

// one token: RMS_NORM -> MUL_MAT -> ADD -> VIEW -> SIGMOID -> VIEW(pre) -> DSV4_HC_PRE -> RMS_NORM -> MUL in one kernel
// (the two functions above chained through a block counter); other shapes run the two functions above
void ggml_cuda_op_rms_norm_mat_vec_pre_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_mat, ggml_tensor * add, ggml_tensor * sig,
        const ggml_tensor * hc_pre, const ggml_tensor * rms_norm_pre, ggml_tensor * mul,
        void * dst_q8_1 = nullptr, int64_t ncols_padded = 0);

// the comb input of DSV4_HC_POST computed from the mix vector:
// e = MUL_MAT(K, mixes); comb = RESHAPE(e) / (scale*POOL_1D_MAX(e*r) + bias) + c0
struct ggml_cuda_hc_select_args {
    const ggml_tensor * K     = nullptr;
    const ggml_tensor * mixes = nullptr;
    const ggml_tensor * r     = nullptr;
    const ggml_tensor * c0    = nullptr;
    float scale = 1.0f;
    float bias  = 0.0f;
};

// DSV4_HC_POST with its comb input computed in the kernel, and optionally the sum of its hc output streams
// (sum != nullptr: the VIEW x hc -> ADD x (hc - 1) chain).
// sum_q8_1 (one token): also writes the sum quantized to q8_1 for mul_mat_vec_q, zero padded to ncols_padded
void ggml_cuda_op_hc_post_fused(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post,
        const ggml_cuda_hc_select_args & sel, ggml_tensor * sum, void * sum_q8_1 = nullptr, int64_t ncols_padded = 0);

// RMS_NORM(g) -> MUL(w) -> RESHAPE(n, 1, rows) -> ADD(streams [n, hc, rows], .): the normed row added to every stream
void ggml_cuda_op_rms_norm_mul_add_streams(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul, ggml_tensor * add);
