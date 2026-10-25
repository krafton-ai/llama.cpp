#include "raon-situ.cuh"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static __global__ void raon_situ_f32(const float * gate, const float * up, float * dst, int64_t n_values) {
    ggml_cuda_pdl_lc();
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_values) {
        return;
    }
    ggml_cuda_pdl_sync();

    const float g = gate[i];
    const float u = up[i];
    const float g_soft = tanhf(g * 0.25f) * 4.0f;
    const float g_sigmoid = 1.0f / (1.0f + expf(-g));
    const float g_act = g_soft * g_sigmoid;
    const float u_soft = tanhf(u * (1.0f / 25.0f)) * 25.0f;
    dst[i] = g_act * u_soft;
}

void ggml_cuda_op_raon_situ(ggml_backend_cuda_context & ctx,
        const ggml_tensor * gate, const ggml_tensor * up, ggml_tensor * dst) {
    const int64_t n_values = ggml_nelements(dst);
    const int block_size = 256;
    const dim3 grid((n_values + block_size - 1) / block_size, 1, 1);
    const ggml_cuda_kernel_launch_params params(grid, dim3(block_size, 1, 1), 0, ctx.stream());
    ggml_cuda_kernel_launch(raon_situ_f32, params,
            (const float *) gate->data, (const float *) up->data, (float *) dst->data, n_values);

    static const bool debug = [] {
        const char * value = getenv("GGML_CUDA_RAON_SITU_DEBUG");
        return value != nullptr && strcmp(value, "1") == 0;
    }();
    if (debug) {
        fprintf(stderr, "RAON_SITU_FUSION ne=[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "] name=%s\n",
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], dst->name);
    }
}
