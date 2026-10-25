#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_dsv4_hc_comb(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_raon_hc_comb(ggml_backend_cuda_context & ctx,
        const ggml_tensor * e, const ggml_tensor * scale, const ggml_tensor * chart,
        const ggml_tensor * identity, ggml_tensor * dst);
void ggml_cuda_op_raon_hc_affine(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mixes, const ggml_tensor * scale, const ggml_tensor * base,
        ggml_tensor * dst, int kind);
void ggml_cuda_op_dsv4_hc_pre(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
// gate_scale_in/gate_scale_out: the SCALE nodes of a fused SCALE -> SIGMOID -> SCALE chain feeding src[2]
void ggml_cuda_op_dsv4_hc_post(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
        const ggml_tensor * gate_scale_in = nullptr, const ggml_tensor * gate_scale_out = nullptr);

void ggml_cuda_op_raon_hc_rms_tail(ggml_backend_cuda_context & ctx,
        const ggml_tensor * mean, const ggml_tensor * selector, ggml_tensor * dst, float eps);

void ggml_cuda_op_raon_hc_helmert(ggml_backend_cuda_context & ctx,
        const ggml_tensor * h, const ggml_tensor * z, ggml_tensor * dst);
