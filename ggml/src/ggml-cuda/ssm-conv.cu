#include "common.cuh"
#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * src0_ptr, const float * src1_ptr,
                                    const float * bias_ptr,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * dst_ptr, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const float * GGML_CUDA_RESTRICT src1 = src1_ptr;
    const float * GGML_CUDA_RESTRICT bias = bias_ptr;
    float       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

    ggml_cuda_pdl_sync();
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(blocks, threads, 0, stream);
            ggml_cuda_kernel_launch(ssm_conv_f32<apply_silu, threads, kNC>, launch_params, src0, src1, bias, src0_nb0, src0_nb1,
                                                                        src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3:  launch_kernel(std::integral_constant<int, 3 >{}); break;
        case 4:  launch_kernel(std::integral_constant<int, 4 >{}); break;
        case 5:  launch_kernel(std::integral_constant<int, 5 >{}); break;
        case 9:  launch_kernel(std::integral_constant<int, 9 >{}); break;
        case 15: launch_kernel(std::integral_constant<int, 15>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9, 15 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// one thread per channel; with fuse_l2 one block per L2 row of the output. The thread reads the old state of its
// channel before it writes any copy, so the cache row it reads may be one that it writes
template <int nc, bool fuse_l2>
static __global__ void conv_state_f32(const ggml_cuda_conv_state_args a) {
    const int64_t c     = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    const bool    valid = c < a.n_ch;

    const float * state = a.state;
    if (a.state_row != nullptr) {
        state += (int64_t) a.state_row[0]*a.state_row_stride;
    }

    float st[nc - 1];
    float w[nc];
#pragma unroll
    for (int j = 0; j < nc - 1; ++j) {
        st[j] = valid ? state[c*a.state_stride_c + j] : 0.0f;
    }
#pragma unroll
    for (int j = 0; j < nc; ++j) {
        w[j] = valid ? a.w[c*a.w_stride_c + j] : 0.0f;
    }

    float win[nc];
#pragma unroll
    for (int j = 0; j < nc - 1; ++j) {
        win[j] = st[j];
    }

    for (int t = 0; t < a.n_t; ++t) {
        win[nc - 1] = valid ? a.x[t*a.x_stride_t + c*a.x_stride_c] : 0.0f;

        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < nc; ++j) {
            sumf += win[j]*w[j];
        }
        const float out = ggml_cuda_op_silu_single(sumf);
        if (valid) {
            a.y[t*a.y_stride_t + c] = out;
        }

        if constexpr (fuse_l2) {
            __shared__ float s_sum[32];
            if ((int64_t) blockIdx.x*blockDim.x < a.l2_nc) { // the same for the whole block
                float sum = warp_reduce_sum(valid ? out*out : 0.0f);
                if (threadIdx.x % WARP_SIZE == 0) {
                    s_sum[threadIdx.x / WARP_SIZE] = sum;
                }
                __syncthreads();
                const int lane = threadIdx.x % WARP_SIZE;
                sum = lane < (int) (blockDim.x / WARP_SIZE) ? s_sum[lane] : 0.0f;
                sum = warp_reduce_sum(sum);
                __syncthreads();
                if (valid) {
                    a.y_l2[t*a.y_l2_stride_t + c] = out*rsqrtf(fmaxf(sum, a.l2_eps*a.l2_eps));
                }
            }
        }

#pragma unroll
        for (int j = 0; j < nc - 1; ++j) {
            win[j] = win[j + 1];
        }
    }

    if (!valid) {
        return;
    }
    for (int k = 0; k < a.n_copies; ++k) {
#pragma unroll
        for (int j = 0; j < nc - 1; ++j) {
            const int p = a.copy_start[k] + j;
            a.copy_dst[k][c*(nc - 1) + j] = p < nc - 1 ? st[p] : a.x[(p - (nc - 1))*a.x_stride_t + c*a.x_stride_c];
        }
    }
}

// the same for many tokens, as ssm_conv_long_token_f32 with the window read from the state and x: one block per
// threads channels and split_n_t tokens, one thread per channel. Only the blocks of the first tokens read the state,
// and they write all the copies after the thread has read its channel, so the cache row read may be one that is written
template <int nc, int threads, int split_n_t>
static __global__ void conv_state_long_f32(const ggml_cuda_conv_state_args a) {
    constexpr int n_cols = nc - 1 + split_n_t;
    __shared__ float smem[threads*n_cols];

    const int     tid   = threadIdx.x;
    const int64_t c     = (int64_t) blockIdx.x*threads + tid;
    const int     p0    = blockIdx.y*split_n_t;
    const bool    valid = c < a.n_ch;
    if (!valid) {
        return;
    }

    const float * state = a.state;
    if (a.state_row != nullptr) {
        state += (int64_t) a.state_row[0]*a.state_row_stride;
    }

    // column p of [state | x] is smem[tid*n_cols + p - p0]; with x transposed the loads are contiguous over channels
    float * win = smem + tid*n_cols;
#pragma unroll
    for (int col = 0; col < n_cols; ++col) {
        const int p = p0 + col;
        float v = 0.0f;
        if (p < nc - 1) {
            v = state[c*a.state_stride_c + p];
        } else if (p - (nc - 1) < a.n_t) {
            v = a.x[(int64_t) (p - (nc - 1))*a.x_stride_t + c*a.x_stride_c];
        }
        win[col] = v;
    }

    float w[nc];
#pragma unroll
    for (int j = 0; j < nc; ++j) {
        w[j] = a.w[c*a.w_stride_c + j];
    }

    const int local_n_t = min(split_n_t, a.n_t - p0);
    for (int i = 0; i < local_n_t; ++i) {
        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < nc; ++j) {
            sumf += win[i + j]*w[j];
        }
        a.y[(int64_t) (p0 + i)*a.y_stride_t + c] = ggml_cuda_op_silu_single(sumf);
    }

    if (blockIdx.y != 0) {
        return;
    }
    for (int k = 0; k < a.n_copies; ++k) {
#pragma unroll
        for (int j = 0; j < nc - 1; ++j) {
            const int p = a.copy_start[k] + j;
            a.copy_dst[k][c*(nc - 1) + j] = p < n_cols ? win[p] : a.x[(int64_t) (p - (nc - 1))*a.x_stride_t + c*a.x_stride_c];
        }
    }
}

void ggml_cuda_op_conv_state(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_args & args) {
    const bool   fuse_l2 = args.y_l2 != nullptr;
    cudaStream_t stream  = ctx.stream();
    GGML_ASSERT(!fuse_l2 || (args.l2_n % WARP_SIZE == 0 && args.l2_n <= 32*WARP_SIZE));
    GGML_ASSERT(!fuse_l2 || args.n_t <= GGML_CUDA_CONV_STATE_SHORT_MAX_TOKENS);

    if (args.n_t > GGML_CUDA_CONV_STATE_SHORT_MAX_TOKENS) {
        constexpr int threads   = 128;
        constexpr int split_n_t = 32;
        const dim3    blocks((args.n_ch + threads - 1) / threads, (args.n_t + split_n_t - 1) / split_n_t, 1);
        switch (args.nc) {
            case 3: conv_state_long_f32<3, threads, split_n_t><<<blocks, threads, 0, stream>>>(args); break;
            case 4: conv_state_long_f32<4, threads, split_n_t><<<blocks, threads, 0, stream>>>(args); break;
            case 5: conv_state_long_f32<5, threads, split_n_t><<<blocks, threads, 0, stream>>>(args); break;
            default: GGML_ABORT("unsupported conv width %d", args.nc);
        }
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    const int  threads = fuse_l2 ? args.l2_n : 128;
    const dim3 blocks((args.n_ch + threads - 1) / threads, 1, 1);

    auto launch = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (fuse_l2) {
            conv_state_f32<kNC, true><<<blocks, threads, 0, stream>>>(args);
        } else {
            conv_state_f32<kNC, false><<<blocks, threads, 0, stream>>>(args);
        }
    };

    switch (args.nc) {
        case 3: launch(std::integral_constant<int, 3>{}); break;
        case 4: launch(std::integral_constant<int, 4>{}); break;
        case 5: launch(std::integral_constant<int, 5>{}); break;
        default: GGML_ABORT("unsupported conv width %d", args.nc);
    }
    CUDA_CHECK(cudaGetLastError());
}

// ---------------------------------------------------------------------------------------------------------------------
// one token of a gated delta net layer with a write/erase ratio: the conv state update of
// ggml_cuda_op_conv_state (with the L2 norm of the q and k heads) and the two gates in one launch.
// Blocks below conv_blocks: conv_state_f32<nc, true> for one head of l2_n channels; the v heads also scale their outputs
// by sigmoid(a)/sigmoid(b) of their head (as sigmoid_ratio_mul_f32) and write sigmoid(b).
// The other blocks: two rows of the f_b product per block, 64 threads per row, the sum of each row as mul_mat_vec_f
// with a block of 64 threads sums it (one float2 per thread, warp sums, then the sum of the two warp sums), then the
// decay gate scale*sigmoid(row + bias) + offset (as add_sigmoid_scale_f32)

static __device__ __forceinline__ float rda_pre_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x)); // op_sigmoid of unary.cu
}

template <int nc>
static __global__ void rda_pre_f32(const ggml_cuda_conv_state_args a, const ggml_cuda_rda_gates_args g) {
    if ((int) blockIdx.x < g.conv_blocks) {
        const int64_t c     = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
        const bool    valid = c < a.n_ch;

        const float * state = a.state;
        if (a.state_row != nullptr) {
            state += (int64_t) a.state_row[0]*a.state_row_stride;
        }

        float st[nc - 1];
        float w[nc];
#pragma unroll
        for (int j = 0; j < nc - 1; ++j) {
            st[j] = valid ? state[c*a.state_stride_c + j] : 0.0f;
        }
#pragma unroll
        for (int j = 0; j < nc; ++j) {
            w[j] = valid ? a.w[c*a.w_stride_c + j] : 0.0f;
        }

        float win[nc];
#pragma unroll
        for (int j = 0; j < nc - 1; ++j) {
            win[j] = st[j];
        }
        win[nc - 1] = valid ? a.x[c*a.x_stride_c] : 0.0f;

        float sumf = 0.0f;
#pragma unroll
        for (int j = 0; j < nc; ++j) {
            sumf += win[j]*w[j];
        }
        const float out = ggml_cuda_op_silu_single(sumf);
        if (valid) {
            a.y[c] = out;
        }

        __shared__ float s_sum[32];
        if ((int64_t) blockIdx.x*blockDim.x < a.l2_nc) { // the same for the whole block
            float sum = warp_reduce_sum(valid ? out*out : 0.0f);
            if (threadIdx.x % WARP_SIZE == 0) {
                s_sum[threadIdx.x / WARP_SIZE] = sum;
            }
            __syncthreads();
            const int lane = threadIdx.x % WARP_SIZE;
            sum = lane < (int) (blockDim.x / WARP_SIZE) ? s_sum[lane] : 0.0f;
            sum = warp_reduce_sum(sum);
            __syncthreads();
            if (valid) {
                a.y_l2[c] = out*rsqrtf(fmaxf(sum, a.l2_eps*a.l2_eps));
            }
        }

        // the v heads: v*(sigmoid(a)/sigmoid(b)) of the head, and sigmoid(b)
        const int64_t cv = c - g.v_off;
        if (valid && cv >= 0 && cv < (int64_t) g.head*g.n_head) {
            const int r  = (int) (cv / g.head);
            const int i0 = (int) (cv - (int64_t) r*g.head);
            const float sa = rda_pre_sigmoid(g.a[r]);
            const float sb = rda_pre_sigmoid(g.b[r]);
            g.v_dst[cv] = out*(sa/sb);
            if (i0 == 0) {
                g.sig_b[r] = sb;
            }
        }

        if (!valid) {
            return;
        }
        for (int k = 0; k < a.n_copies; ++k) {
#pragma unroll
            for (int j = 0; j < nc - 1; ++j) {
                const int p = a.copy_start[k] + j;
                a.copy_dst[k][c*(nc - 1) + j] = p < nc - 1 ? st[p] : a.x[(p - (nc - 1))*a.x_stride_t + c*a.x_stride_c];
            }
        }
        return;
    }

    // f_b rows and the decay gate
    __shared__ float s_w[4];
    const int hb   = threadIdx.x / 64; // which of the two rows of the block
    const int tid  = threadIdx.x % 64;
    const int row  = 2*((int) blockIdx.x - g.conv_blocks) + hb;
    float sum = 0.0f;
    if (row < g.d_inner && tid < g.rank/2) {
        const float2 y2 = ((const float2 *) g.fa)[tid];
        if (g.fb_type == 0) {
            const float2 x2 = ((const float2 *) ((const float *) g.fb_w + (int64_t) row*g.fb_stride_row))[tid];
            ggml_cuda_mad(sum, x2.x, y2.x);
            ggml_cuda_mad(sum, x2.y, y2.y);
        } else if (g.fb_type == 1) {
            const float2 x2 = __half22float2(((const half2 *) ((const half *) g.fb_w + (int64_t) row*g.fb_stride_row))[tid]);
            ggml_cuda_mad(sum, x2.x, y2.x);
            ggml_cuda_mad(sum, x2.y, y2.y);
        } else {
#ifdef FP16_AVAILABLE
            const half2 x2 = ((const half2 *) ((const half *) g.fb_w + (int64_t) row*g.fb_stride_row))[tid];
            half2 sumh2 = make_half2(0.0f, 0.0f);
            sumh2 += x2 * make_half2(y2.x, y2.y);
            sum = __low2float(sumh2) + __high2float(sumh2);
#else
            NO_DEVICE_CODE;
#endif // FP16_AVAILABLE
        }
    }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x % WARP_SIZE == 0) {
        s_w[threadIdx.x / WARP_SIZE] = sum;
    }
    __syncthreads();
    if (tid == 0 && row < g.d_inner) {
        const float y = s_w[2*hb] + s_w[2*hb + 1];
        if (g.fb_out) {
            g.fb_out[row] = y;
        }
        const float v = y + g.dt_b[row % g.bias_n];
        g.g_out[row] = g.scale*rda_pre_sigmoid(v) + g.offset;
    }
}

void ggml_cuda_op_rda_pre(ggml_backend_cuda_context & ctx, const ggml_cuda_conv_state_args & args,
        ggml_cuda_rda_gates_args gates) {
    GGML_ASSERT(args.n_t == 1 && args.y_l2 != nullptr && args.l2_n == 128 && gates.rank <= 128 && gates.rank % 2 == 0);
    gates.conv_blocks = (int) ((args.n_ch + args.l2_n - 1) / args.l2_n);
    const dim3 blocks(gates.conv_blocks + (gates.d_inner + 1)/2, 1, 1);
    cudaStream_t stream = ctx.stream();
    switch (args.nc) {
        case 3: rda_pre_f32<3><<<blocks, 128, 0, stream>>>(args, gates); break;
        case 4: rda_pre_f32<4><<<blocks, 128, 0, stream>>>(args, gates); break;
        case 5: rda_pre_f32<5><<<blocks, 128, 0, stream>>>(args, gates); break;
        default: GGML_ABORT("unsupported conv width %d", args.nc);
    }
    CUDA_CHECK(cudaGetLastError());
}
