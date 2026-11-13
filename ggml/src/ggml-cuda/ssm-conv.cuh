#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// CONCAT(state, x) -> CPY(columns of the concat -> state cache) -> SSM_CONV -> SILU [-> L2_NORM of rows of the output]
// for one sequence, as one kernel (see ggml_cuda_match_conv_state in ggml-cuda.cu). Above SHORT_MAX_TOKENS tokens a
// kernel tiled over tokens runs, without the L2_NORM
#define GGML_CUDA_CONV_STATE_MAX_COPIES 8
#define GGML_CUDA_CONV_STATE_SHORT_MAX_TOKENS 32

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

// one token of a gated delta net layer with a write/erase ratio: ggml_cuda_op_conv_state (with the L2 norm, l2_n == 128)
// and the two gates (sigmoid_ratio_mul of the v heads, the f_b product and the decay gate) in one launch
struct ggml_cuda_rda_gates_args {
    const float * a;        // a[r], b[r]: the logits of the two ratio sigmoids of head r
    const float * b;
    float       * sig_b;    // sigmoid(b), [n_head]
    float       * v_dst;    // v*(sigmoid(a)/sigmoid(b)), [head, n_head]
    int64_t       v_off;    // channel of v[0] in the conv output
    int           head;     // v channels per head (== l2_n)
    int           n_head;
    const void  * fb_w;     // f_b weights (F32 or F16), row r at element r*fb_stride_row, rank values
    int64_t       fb_stride_row;
    int           fb_type;  // 0: F32, 1: F16 summed in F32, 2: F16 summed in half2 (as mul_mat_vec_f picks for the node)
    const float * fa;       // f_a, rank values
    int           rank;
    float       * fb_out;   // f_b product, [d_inner]
    const float * dt_b;     // bias, element r % bias_n
    int64_t       bias_n;
    float       * g_out;    // scale*sigmoid(fb_out + dt_b) + offset
    float         scale;
    float         offset;
    int           d_inner;
    int           conv_blocks;
};

void ggml_cuda_op_rda_pre(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_args & args,
        ggml_cuda_rda_gates_args gates);
