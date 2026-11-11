#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>

#include "argmax.cuh"
#include "common.cuh"
#include "sum.cuh"

// the largest value, and the smallest index among equal values (deterministic, also across the chunks below)
static __device__ __forceinline__ void argmax_merge(float & maxval, int & argmax, const float val, const int col) {
    if (val > maxval || (val == maxval && col < argmax)) {
        maxval = val;
        argmax = col;
    }
}

// a block spans one row, or with chunk > 0 one chunk of a single row: then it also leaves its value in vals
static __global__ void argmax_f32(
        const float * __restrict__ x, int32_t * __restrict__ dst, float * __restrict__ vals, const int64_t ncols, const int64_t chunk) {
    const int64_t row = blockIdx.x;

    float maxval = -INFINITY;
    int   argmax = INT_MAX;
    const float * rowx = chunk > 0 ? x : x + row * ncols;
    const int32_t col0 = chunk > 0 ? row * chunk : 0;
    const int32_t col1 = chunk > 0 && col0 + chunk < ncols ? col0 + chunk : ncols;

    for (int32_t col = col0 + threadIdx.x; col < col1; col += blockDim.x) {
        argmax_merge(maxval, argmax, rowx[col], col);
    }

#pragma unroll
    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
        const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
        argmax_merge(maxval, argmax, val, col);
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    if (n_warps > 1) {
        constexpr int    max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_maxval[max_warps];
        __shared__ int   shared_argmax[max_warps];
        if (lane_id == 0) {
            shared_maxval[warp_id] = maxval;
            shared_argmax[warp_id] = argmax;
        }

        __syncthreads();

        if (warp_id == 0) {
            if (lane_id < n_warps) {
                maxval = shared_maxval[lane_id];
                argmax = shared_argmax[lane_id];
            }
#pragma unroll
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                const float val = __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE);
                const int   col = __shfl_xor_sync(0xFFFFFFFF, argmax, offset, WARP_SIZE);
                argmax_merge(maxval, argmax, val, col);
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        dst[row] = argmax == INT_MAX ? 0 : argmax; // only NaN in the row
        if (chunk > 0) {
            vals[row] = maxval;
        }
    }
}

static __global__ void argmax_chunks_f32(
        const float * __restrict__ vals, const int32_t * __restrict__ cols, int32_t * __restrict__ dst, const int n) {
    float maxval = vals[0];
    int   argmax = cols[0];
    for (int i = 1; i < n; i++) {
        if (vals[i] > maxval) { // the chunks come in the order of their indices
            maxval = vals[i];
            argmax = cols[i];
        }
    }
    *dst = argmax;
}

void ggml_cuda_argmax(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_I32);

    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ne00  = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const float * src0_d = (const float *) src0->data;
    int32_t     * dst_d  = (int32_t     *) dst->data;

    cudaStream_t stream = ctx.stream();

    // one block runs on one SM: a long row, the logits of a vocabulary, is shared among the SMs in chunks
    constexpr int64_t chunk = 4096;
    if (nrows == 1 && ne00 >= 2 * chunk) {
        const int64_t n_chunks = (ne00 + chunk - 1) / chunk;
        ggml_cuda_pool_alloc<float>   vals(ctx.pool(), n_chunks);
        ggml_cuda_pool_alloc<int32_t> cols(ctx.pool(), n_chunks);
        argmax_f32<<<n_chunks, 256, 0, stream>>>(src0_d, cols.get(), vals.get(), ne00, chunk);
        argmax_chunks_f32<<<1, 1, 0, stream>>>(vals.get(), cols.get(), dst_d, n_chunks);
        return;
    }

    const int64_t num_blocks = nrows;
    const int64_t num_threads = std::min<int64_t>(1024, (ne00 + WARP_SIZE - 1) / WARP_SIZE * WARP_SIZE);
    const dim3 blocks_dim(num_threads, 1, 1);
    const dim3 blocks_num(num_blocks, 1, 1);

    argmax_f32<<<blocks_num, blocks_dim, 0, stream>>>(src0_d, dst_d, nullptr, ne00, 0);
}
