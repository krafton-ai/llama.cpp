#include "common.cuh"

void ggml_cuda_op_raon_situ(ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * dst);
