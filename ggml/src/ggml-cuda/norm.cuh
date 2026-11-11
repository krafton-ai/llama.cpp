#include "common.cuh"

void ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_group_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rms_norm_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor);

void ggml_cuda_op_rms_norm_scale_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * scale_tensor);

void ggml_cuda_op_rms_norm_fused_add(ggml_backend_cuda_context & ctx,
                                     ggml_tensor *               dst,
                                     ggml_tensor *               mul_tensor,
                                     ggml_tensor *               add_tensor);

// MUL(RMS_NORM(x), w) for a single row x with w of the same length; also writes the result quantized to q8_1 (the
// quantization of quantize_row_q8_1_cuda, zero padded to ncols_padded) to dst_q8_1 for mul_mat_vec_q
void ggml_cuda_op_rms_norm_mul_q8_1(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_tensor,
                                    void * dst_q8_1, int64_t ncols_padded);

void ggml_cuda_op_rms_norm_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_l2_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// MUL(MUL(RMS_NORM(x), w), unary(g)) with w broadcast along the rows of x and unary SIGMOID or SILU (a gated RMS norm)
void ggml_cuda_op_rms_norm_mul_gate(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul_w,
                                    ggml_tensor * gate, ggml_tensor * mul_out);
