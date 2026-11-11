#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// CONCAT(state, x) -> CPY(columns of the concat -> state cache) -> SSM_CONV -> SILU [-> L2_NORM of rows of the output]
// for one sequence, as one kernel (see ggml_cuda_match_conv_state in ggml-cuda.cu)
#define GGML_CUDA_CONV_STATE_MAX_COPIES 8

struct ggml_cuda_conv_state_args {
    const float   * state;            // element (j, c) of the old state at state[c*state_stride_c + j]
    const int32_t * state_row;        // if set: the old state is row state_row[0] of the cache at state
    int64_t         state_row_stride; // in floats
    int64_t         state_stride_c;
    const float   * x;                // element (t, c) of the new inputs at x[t*x_stride_t + c*x_stride_c]
    int64_t         x_stride_t;
    int64_t         x_stride_c;
    const float   * w;                // conv weight w[c*w_stride_c + j]
    int64_t         w_stride_c;
    float         * y;                // silu(conv) at y[t*y_stride_t + c]
    int64_t         y_stride_t;
    float         * y_l2;             // if set: each row of l2_n channels below l2_nc divided by its L2 norm (eps l2_eps)
    int64_t         y_l2_stride_t;
    int64_t         l2_nc;
    int             l2_n;
    float           l2_eps;
    int             n_copies;         // copy k writes the nc - 1 columns from copy_start[k] of [state | x] to copy_dst[k]
    float         * copy_dst[GGML_CUDA_CONV_STATE_MAX_COPIES];
    int             copy_start[GGML_CUDA_CONV_STATE_MAX_COPIES];
    int64_t         n_ch;
    int             n_t;
    int             nc;
};

void ggml_cuda_op_conv_state(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_args & args);
