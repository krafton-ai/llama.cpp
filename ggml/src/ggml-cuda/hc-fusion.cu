#include "hc-fusion.cuh"
#include "convert.cuh"

#include <climits>

static constexpr int HC_MAX = 8;

static __device__ __forceinline__ float hc_sigmoid(const float x) {
    return 1.0f/(1.0f + expf(-x));
}

// ---------------------------------------------------------------------------------------------------------------------
// RMS_NORM -> MUL_MAT (few rows) -> ADD [-> SIGMOID]: rms_norm(x) w = (x w) / rms(x), so the partial dot products and
// the partial sums of squares of each input chunk are computed in one pass, then reduced per token.
// part layout: [token][split][nrows_w + 1], the last entry is the sum of squares of the chunk.

static constexpr int RMS_MV_WARPS   = 4;
static constexpr int RMS_MV_TILE    = 1024;
static constexpr int RMS_MV_PP_WARPS = 8;
static constexpr int RMS_MV_PP_ROWS  = 3;   // up to RMS_MV_PP_WARPS*RMS_MV_PP_ROWS - 1 weight rows
static constexpr int RMS_MV_PP_TILE  = 128;

// one block per (split, group of RMS_MV_WARPS rows, group of n_tok tokens), one warp per row (the row after the last
// weight row is the sum of squares). The x tile goes through shared memory once for all rows of the block
template <int n_tok, typename T>
static __global__ void rms_norm_mat_vec_part_f32(
        const float * x, const T * w, float * part, const int nrows_w, const int chunk, const int nt,
        const int64_t sx1, const int64_t sw1) {
    const int s     = blockIdx.x;
    const int k     = blockIdx.y*RMS_MV_WARPS + threadIdx.y;
    const int t0    = blockIdx.z*n_tok;
    const int lane  = threadIdx.x;
    const int tid   = threadIdx.y*WARP_SIZE + lane;
    const int nsplit = gridDim.x;

    __shared__ float xs[n_tok][RMS_MV_TILE];

    ggml_cuda_pdl_sync();

    const T * wr = w + (int64_t) min(k, nrows_w - 1)*sw1 + (int64_t) s*chunk;

    if constexpr (n_tok == 1) {
        // one token: every warp reads x from global memory (cached), two inputs per lane and step
        if (chunk % 2 == 0 && sw1 % 2 == 0) {
            if (k > nrows_w) {
                return;
            }
            const float * xr = x + (int64_t) s*chunk;
            float a = 0.0f;
            for (int i = 2*lane; i < chunk; i += 2*WARP_SIZE) {
                const float2 xv = *(const float2 *) (xr + i);
                float2 wv = xv;
                if (k < nrows_w) {
                    if constexpr (std::is_same_v<T, half>) {
                        wv = __half22float2(*(const half2 *) (wr + i));
                    } else {
                        wv = *(const float2 *) (wr + i);
                    }
                }
                a = fmaf(wv.x, xv.x, a);
                a = fmaf(wv.y, xv.y, a);
            }
            a = warp_reduce_sum(a);
            if (lane == 0) {
                part[(int64_t) s*(nrows_w + 1) + k] = a;
            }
            return;
        }
    }

    float acc[n_tok] = {0.0f};
    for (int i0 = 0; i0 < chunk; i0 += RMS_MV_TILE) {
        const int len = min(RMS_MV_TILE, chunk - i0);
        for (int j = tid; j < n_tok*len; j += RMS_MV_WARPS*WARP_SIZE) {
            const int t = j/len;
            const int i = j - t*len;
            xs[t][i] = t0 + t < nt ? x[(int64_t) (t0 + t)*sx1 + (int64_t) s*chunk + i0 + i] : 0.0f;
        }
        __syncthreads();
        if (k < nrows_w) {
            for (int i = lane; i < len; i += WARP_SIZE) {
                const float wv = ggml_cuda_cast<float>(wr[i0 + i]);
#pragma unroll
                for (int t = 0; t < n_tok; ++t) {
                    acc[t] = fmaf(wv, xs[t][i], acc[t]);
                }
            }
        } else if (k == nrows_w) {
            for (int i = lane; i < len; i += WARP_SIZE) {
#pragma unroll
                for (int t = 0; t < n_tok; ++t) {
                    acc[t] = fmaf(xs[t][i], xs[t][i], acc[t]);
                }
            }
        }
        __syncthreads();
    }

    if (k > nrows_w) {
        return;
    }
#pragma unroll
    for (int t = 0; t < n_tok; ++t) {
        acc[t] = warp_reduce_sum(acc[t]);
        if (lane == 0 && t0 + t < nt) {
            part[((int64_t) (t0 + t)*nsplit + s)*(nrows_w + 1) + k] = acc[t];
        }
    }
}

// many tokens: one block per (split, 32 tokens) covers every row, one lane per token, so x is read once and no
// cross-lane sums are needed. Tiles of x (padded rows: lane t reads row t) and of w go through shared memory
template <typename T>
static __global__ void rms_norm_mat_vec_part_pp_f32(
        const float * x, const T * w, float * part, const int nrows_w, const int chunk, const int nt,
        const int64_t sx1, const int64_t sw1) {
    constexpr int n_rows = RMS_MV_PP_WARPS*RMS_MV_PP_ROWS;
    const int s      = blockIdx.x;
    const int t0     = blockIdx.z*WARP_SIZE;
    const int lane   = threadIdx.x;
    const int tid    = threadIdx.y*WARP_SIZE + lane;
    const int nsplit = gridDim.x;

    // rows padded by 4 floats: lanes read float4 at a stride of 4 banks, no conflicts per quarter warp
    __shared__ __align__(16) float xs[WARP_SIZE][RMS_MV_PP_TILE + 4];
    __shared__ __align__(16) float ws[n_rows][RMS_MV_PP_TILE];

    ggml_cuda_pdl_sync();

    float acc[RMS_MV_PP_ROWS] = {0.0f};
    for (int i0 = 0; i0 < chunk; i0 += RMS_MV_PP_TILE) {
        const int len = min(RMS_MV_PP_TILE, chunk - i0);
        for (int j = tid; j < WARP_SIZE*len; j += RMS_MV_PP_WARPS*WARP_SIZE) {
            const int t = j/len;
            const int i = j - t*len;
            xs[t][i] = t0 + t < nt ? x[(int64_t) (t0 + t)*sx1 + (int64_t) s*chunk + i0 + i] : 0.0f;
        }
        for (int j = tid; j < nrows_w*len; j += RMS_MV_PP_WARPS*WARP_SIZE) {
            const int k = j/len;
            const int i = j - k*len;
            ws[k][i] = ggml_cuda_cast<float>(w[(int64_t) k*sw1 + (int64_t) s*chunk + i0 + i]);
        }
        __syncthreads();
        for (int i = 0; i < len; i += 4) {
            const float4 xv = *(const float4 *) &xs[lane][i];
#pragma unroll
            for (int r = 0; r < RMS_MV_PP_ROWS; ++r) {
                const int k = threadIdx.y + r*RMS_MV_PP_WARPS;
                const float4 wv = k < nrows_w ? *(const float4 *) &ws[k][i] : xv;
                if (k <= nrows_w) {
                    acc[r] = fmaf(wv.x, xv.x, acc[r]);
                    acc[r] = fmaf(wv.y, xv.y, acc[r]);
                    acc[r] = fmaf(wv.z, xv.z, acc[r]);
                    acc[r] = fmaf(wv.w, xv.w, acc[r]);
                }
            }
        }
        __syncthreads();
    }

    if (t0 + lane < nt) {
#pragma unroll
        for (int r = 0; r < RMS_MV_PP_ROWS; ++r) {
            const int k = threadIdx.y + r*RMS_MV_PP_WARPS;
            if (k <= nrows_w) {
                part[((int64_t) (t0 + lane)*nsplit + s)*(nrows_w + 1) + k] = acc[r];
            }
        }
    }
}

// one warp per token: sum the splits, scale by 1/rms, add the bias [and write sigmoid of the first n_sig columns]
template <bool has_sig>
static __global__ void rms_norm_mat_vec_reduce_f32(
        const float * part, const float * bias, float * dst, float * sig,
        const int nrows_w, const int nsplit, const int n, const float eps, const int nt,
        const int64_t sb1, const int64_t sd1, const int64_t ss1, const int n_sig) {
    const int t    = blockIdx.x*RMS_MV_WARPS + threadIdx.y;
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();

    if (t >= nt) {
        return;
    }

    const float * p = part + (int64_t) t*nsplit*(nrows_w + 1);
    float ss = 0.0f;
    for (int s = 0; s < nsplit; ++s) {
        ss += p[s*(nrows_w + 1) + nrows_w];
    }
    const float scale = rsqrtf(ss/n + eps);

    for (int k = lane; k < nrows_w; k += WARP_SIZE) {
        float v = 0.0f;
        for (int s = 0; s < nsplit; ++s) {
            v += p[s*(nrows_w + 1) + k];
        }
        const float y = v*scale + bias[k + t*sb1];
        dst[k + t*sd1] = y;
        if constexpr (has_sig) {
            if (k < n_sig) {
                sig[k + t*ss1] = hc_sigmoid(y);
            }
        }
    }
}

template <typename T>
static void rms_norm_mat_vec_part_launch(
        const float * x, const T * w, float * part, const int nrows_w, const int nsplit, const int chunk, const int nt,
        const int64_t sx1, const int64_t sw1, cudaStream_t stream) {
    if (nt > 1 && nrows_w + 1 <= RMS_MV_PP_WARPS*RMS_MV_PP_ROWS && chunk % 4 == 0) {
        const dim3 grid(nsplit, 1, (nt + WARP_SIZE - 1)/WARP_SIZE);
        const ggml_cuda_kernel_launch_params lp(grid, dim3(WARP_SIZE, RMS_MV_PP_WARPS), 0, stream);
        ggml_cuda_kernel_launch(rms_norm_mat_vec_part_pp_f32<T>, lp, x, w, part, nrows_w, chunk, nt, sx1, sw1);
        return;
    }
    const int n_row_groups = (nrows_w + 1 + RMS_MV_WARPS - 1)/RMS_MV_WARPS;
    if (nt == 1) {
        const ggml_cuda_kernel_launch_params lp(dim3(nsplit, n_row_groups, 1), dim3(WARP_SIZE, RMS_MV_WARPS), 0, stream);
        ggml_cuda_kernel_launch(rms_norm_mat_vec_part_f32<1, T>, lp, x, w, part, nrows_w, chunk, nt, sx1, sw1);
    } else {
        const ggml_cuda_kernel_launch_params lp(dim3(nsplit, n_row_groups, (nt + 7)/8), dim3(WARP_SIZE, RMS_MV_WARPS), 0, stream);
        ggml_cuda_kernel_launch(rms_norm_mat_vec_part_f32<8, T>, lp, x, w, part, nrows_w, chunk, nt, sx1, sw1);
    }
}

void ggml_cuda_op_rms_norm_mat_vec_add(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_mat, ggml_tensor * add, ggml_tensor * sig) {
    const ggml_tensor * x    = rms_norm->src[0];
    const ggml_tensor * w    = mul_mat->src[0];
    const ggml_tensor * bias = add->src[0] == mul_mat ? add->src[1] : add->src[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && ggml_is_contiguous(x));
    GGML_ASSERT(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16);
    GGML_ASSERT(bias->type == GGML_TYPE_F32 && bias->nb[0] == sizeof(float));
    GGML_ASSERT(add->type == GGML_TYPE_F32 && ggml_is_contiguous(add));

    const int     n       = x->ne[0];
    const int     nt      = ggml_nrows(x);
    const int     nrows_w = w->ne[1];
    const float   eps     = ggml_get_op_params_f32(rms_norm, 0);
    const int64_t sw1     = w->nb[1]/ggml_element_size(w);

    // split the input so that a few tokens still fill the GPU; every chunk keeps a multiple of 4 inputs.
    // The many-token kernel has only one block per split and 32 tokens, so it keeps 16 splits at any token count
    const bool pp_rows = nrows_w + 1 <= RMS_MV_PP_WARPS*RMS_MV_PP_ROWS;
    int nsplit = nt <= 8 || pp_rows ? 16 : nt <= 64 ? 4 : nt <= 256 ? 2 : 1;
    while (nsplit > 1 && (n % (4*nsplit) != 0 || n/nsplit < 256)) {
        nsplit /= 2;
    }
    const int chunk = n/nsplit;

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float> part(ctx.pool(), (size_t) nt*nsplit*(nrows_w + 1));

    if (w->type == GGML_TYPE_F16) {
        rms_norm_mat_vec_part_launch(
                (const float *) x->data, (const half *) w->data, part.get(), nrows_w, nsplit, chunk, nt, n, sw1, stream);
    } else {
        rms_norm_mat_vec_part_launch(
                (const float *) x->data, (const float *) w->data, part.get(), nrows_w, nsplit, chunk, nt, n, sw1, stream);
    }

    const int64_t sb1 = ggml_nrows(bias) == 1 ? 0 : bias->nb[1]/sizeof(float);
    const int64_t sd1 = add->nb[1]/sizeof(float);
    const ggml_cuda_kernel_launch_params lp(dim3((nt + RMS_MV_WARPS - 1)/RMS_MV_WARPS), dim3(WARP_SIZE, RMS_MV_WARPS), 0, stream);
    if (sig) {
        ggml_cuda_kernel_launch(rms_norm_mat_vec_reduce_f32<true>, lp, (const float *) part.get(), (const float *) bias->data,
                (float *) add->data, (float *) sig->data, nrows_w, nsplit, n, eps, nt, sb1, sd1,
                (int64_t) (sig->nb[1]/sizeof(float)), (int) sig->ne[0]);
    } else {
        ggml_cuda_kernel_launch(rms_norm_mat_vec_reduce_f32<false>, lp, (const float *) part.get(), (const float *) bias->data,
                (float *) add->data, (float *) nullptr, nrows_w, nsplit, n, eps, nt, sb1, sd1, (int64_t) 0, 0);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// DSV4_HC_PRE -> RMS_NORM -> MUL: one block per token

// token t of x [n, hc, tokens]: dst = rms_norm(scale_pre * sum_s x(., s) pre(s)) * w, one 1-D block of block_size threads.
// coherent: pre was written by this block in the same kernel, read it from L2.
// q8 != nullptr (one token): dst is also written quantized to q8_1, zero padded to n_padded (a multiple of QK8_1)
template <int block_size, bool coherent>
static __device__ __forceinline__ void hc_pre_rms_norm_mul_row(
        const float * x, const float * pre, const float * w, float * dst,
        const int n, const int hc, const float scale_pre, const float eps,
        const int64_t sx1, const int64_t sx2, const int64_t sp0, const int64_t sp1, const int64_t sd1, const int t,
        float * s_pre, float * s_sum, block_q8_1 * q8, const int n_padded) {
    static_assert(block_size % WARP_SIZE == 0, "whole warps");
    if ((int) threadIdx.x < hc) {
        const float * p = pre + threadIdx.x*sp0 + t*sp1;
        if constexpr (coherent) {
            s_pre[threadIdx.x] = __ldcg(p);
        } else {
            s_pre[threadIdx.x] = *p;
        }
    }
    __syncthreads();

    const float * xt = x + t*sx2;
    float * y = dst + t*sd1;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += block_size) {
        float h = 0.0f;
        for (int s = 0; s < hc; ++s) {
            h += xt[i + s*sx1]*s_pre[s];
        }
        h *= scale_pre;
        y[i] = h;
        ss += h*h;
    }
    ss = block_reduce<block_reduce_method::SUM, block_size>(ss, s_sum);
    const float scale = rsqrtf(ss/n + eps);
    // every thread rereads only the elements it wrote
    if (q8 == nullptr) {
        for (int i = threadIdx.x; i < n; i += block_size) {
            y[i] = y[i]*scale*w[i];
        }
        return;
    }
    // n_padded is a multiple of WARP_SIZE: the loop condition is the same for all lanes of a warp
    for (int i = threadIdx.x; i < n_padded; i += block_size) {
        float v = 0.0f;
        if (i < n) {
            v = y[i]*scale*w[i];
            y[i] = v;
        }
        quantize_q8_1_store(v, q8, i);
    }
}

// the same for one token with one float4 per thread, kept in registers between the sum of squares and the write:
// 4*blockDim.x >= n_padded (or >= n without q8), n % 4 == 0, 16-byte aligned x rows, w and dst. A q8_1 block of the copy
// is 8 lanes. pre was written by this block in the same kernel: read it from L2
static __device__ __forceinline__ void hc_pre_rms_norm_mul_row_vec4(
        const float * x, const float * pre, const float * w, float * dst, const int n, const int hc, const float scale_pre,
        const float eps, const int64_t sx1, const int64_t sp0, float * s_pre, float * s_sum, block_q8_1 * q8, const int n_padded) {
    if ((int) threadIdx.x < hc) {
        s_pre[threadIdx.x] = __ldcg(pre + threadIdx.x*sp0);
    }
    __syncthreads();

    const int c = 4*threadIdx.x;
    float4 h = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (c < n) {
        for (int s = 0; s < hc; ++s) {
            const float4 v = *(const float4 *) (x + s*sx1 + c);
            const float  p = s_pre[s];
            h.x = fmaf(v.x, p, h.x);
            h.y = fmaf(v.y, p, h.y);
            h.z = fmaf(v.z, p, h.z);
            h.w = fmaf(v.w, p, h.w);
        }
        h = make_float4(h.x*scale_pre, h.y*scale_pre, h.z*scale_pre, h.w*scale_pre);
    }
    float ss = h.x*h.x + h.y*h.y + h.z*h.z + h.w*h.w;
    ss = block_reduce<block_reduce_method::SUM>(ss, s_sum);
    const float scale = rsqrtf(ss/n + eps);

    float4 o = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (c < n) {
        const float4 wv = *(const float4 *) (w + c);
        o = make_float4(h.x*scale*wv.x, h.y*scale*wv.y, h.z*scale*wv.z, h.w*scale*wv.w);
        *(float4 *) (dst + c) = o;
    }
    if (q8 == nullptr) {
        return;
    }
    // all lanes take part in the reductions; the zero padding up to n_padded comes from the threads past n
    const float amax = warp_reduce_max<QK8_1/4>(fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w))));
    const float sum  = warp_reduce_sum<QK8_1/4>(o.x + o.y + o.z + o.w);
    if (c < n_padded) {
        const float  d = amax / 127.0f;
        block_q8_1 & b = q8[c / QK8_1];
        char4 qv = make_char4(0, 0, 0, 0);
        if (amax != 0.0f) {
            qv = make_char4(roundf(o.x/d), roundf(o.y/d), roundf(o.z/d), roundf(o.w/d));
        }
        *(char4 *) (b.qs + c % QK8_1) = qv;
        if (c % QK8_1 == 0) {
            b.ds = make_half2(d, sum);
        }
    }
}

template <int block_size>
static __global__ void hc_pre_rms_norm_mul_f32(
        const float * x, const float * pre, const float * w, float * dst,
        const int n, const int hc, const float scale_pre, const float eps,
        const int64_t sx1, const int64_t sx2, const int64_t sp0, const int64_t sp1, const int64_t sd1,
        block_q8_1 * q8, const int n_padded) {
    __shared__ float s_pre[HC_MAX];
    __shared__ float s_sum[32];

    ggml_cuda_pdl_sync();

    hc_pre_rms_norm_mul_row<block_size, false>(x, pre, w, dst, n, hc, scale_pre, eps, sx1, sx2, sp0, sp1, sd1,
            blockIdx.x, s_pre, s_sum, q8, n_padded);
}

void ggml_cuda_op_hc_pre_rms_norm_mul(ggml_backend_cuda_context & ctx, const ggml_tensor * hc_pre,
        const ggml_tensor * rms_norm, ggml_tensor * mul, void * dst_q8_1, const int64_t ncols_padded) {
    const ggml_tensor * x   = hc_pre->src[0];
    const ggml_tensor * pre = hc_pre->src[1];
    const ggml_tensor * w   = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float));
    GGML_ASSERT(pre->type == GGML_TYPE_F32);
    GGML_ASSERT(w->type == GGML_TYPE_F32 && ggml_is_contiguous(w));
    GGML_ASSERT(mul->type == GGML_TYPE_F32 && mul->nb[0] == sizeof(float));
    GGML_ASSERT(x->ne[1] <= HC_MAX);

    const int n  = x->ne[0];
    const int hc = x->ne[1];
    const int nt = x->ne[2];
    GGML_ASSERT(dst_q8_1 == nullptr || (nt == 1 && ncols_padded >= n && ncols_padded % QK8_1 == 0 && ncols_padded <= INT_MAX));

    constexpr int block_size = 256;
    const ggml_cuda_kernel_launch_params lp(dim3(nt), dim3(block_size), 0, ctx.stream());
    ggml_cuda_kernel_launch(hc_pre_rms_norm_mul_f32<block_size>, lp,
            (const float *) x->data, (const float *) pre->data, (const float *) w->data, (float *) mul->data,
            n, hc, ggml_get_op_params_f32(hc_pre, 0), ggml_get_op_params_f32(rms_norm, 0),
            (int64_t) (x->nb[1]/sizeof(float)), (int64_t) (x->nb[2]/sizeof(float)),
            (int64_t) (pre->nb[0]/sizeof(float)), (int64_t) (pre->nb[1]/sizeof(float)),
            (int64_t) (mul->nb[1]/sizeof(float)), (block_q8_1 *) dst_q8_1, (int) ncols_padded);
}

// ---------------------------------------------------------------------------------------------------------------------
// one token: RMS_NORM -> MUL_MAT -> ADD -> VIEW -> SIGMOID -> VIEW(pre) -> DSV4_HC_PRE -> RMS_NORM -> MUL in one kernel.
// Every block computes the partial dot products of one split and one group of rows (as rms_norm_mat_vec_part_f32<1>),
// then the last block to finish (block counter, zero before the launch and set back to zero by that block) reduces
// them into the mixes and the sigmoid
// (as rms_norm_mat_vec_reduce_f32) and runs the pre-norm row (as hc_pre_rms_norm_mul_f32).

static constexpr int HC_CHAIN_BLOCK = 256;

struct hc_chain_pre_norm_params {
    const float * x;        // DSV4_HC_PRE input [n, hc]
    const float * pre;      // view of the sigmoid output
    const float * w;        // norm weight
    float       * dst;      // MUL
    int     n;
    int     hc;
    float   scale_pre;
    float   eps;
    int64_t sx1;
    int64_t sp0;
    block_q8_1 * q8;        // q8_1 copy of dst or nullptr
    int     n_padded;
};

// vec4: blockDim.x threads as hc_pre_rms_norm_mul_row_vec4 needs them, else HC_CHAIN_BLOCK
template <typename T, bool vec4>
static __global__ void rms_norm_mat_vec_pre_norm_f32(
        const float * x, const T * w, float * part, unsigned int * counter, const int nrows_w, const int chunk,
        const int64_t sw1, const float * bias, float * mix, float * sig, const int n_sig, const int n_in, const float eps,
        const hc_chain_pre_norm_params pn) {
    const int n_warps = blockDim.x/WARP_SIZE;
    const int s      = blockIdx.x;
    const int lane   = threadIdx.x % WARP_SIZE;
    const int k      = blockIdx.y*n_warps + threadIdx.x/WARP_SIZE;
    const int nsplit = gridDim.x;

    __shared__ bool  s_last;
    __shared__ float s_pre[HC_MAX];
    __shared__ float s_sum[32];

    ggml_cuda_pdl_sync();

    if (k <= nrows_w) {
        const T     * wr = w + (int64_t) min(k, nrows_w - 1)*sw1 + (int64_t) s*chunk;
        const float * xr = x + (int64_t) s*chunk;
        float a = 0.0f;
        for (int i = 2*lane; i < chunk; i += 2*WARP_SIZE) {
            const float2 xv = *(const float2 *) (xr + i);
            float2 wv = xv;
            if (k < nrows_w) {
                if constexpr (std::is_same_v<T, half>) {
                    wv = __half22float2(*(const half2 *) (wr + i));
                } else {
                    wv = *(const float2 *) (wr + i);
                }
            }
            a = fmaf(wv.x, xv.x, a);
            a = fmaf(wv.y, xv.y, a);
        }
        a = warp_reduce_sum(a);
        if (lane == 0) {
            part[(int64_t) s*(nrows_w + 1) + k] = a;
            __threadfence();
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        s_last = atomicAdd(counter, 1u) == gridDim.x*gridDim.y - 1;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    __threadfence();
    if (threadIdx.x == 0) {
        *counter = 0;
    }

    float ss = 0.0f;
    for (int i = 0; i < nsplit; ++i) {
        ss += __ldcg(part + i*(nrows_w + 1) + nrows_w);
    }
    const float scale = rsqrtf(ss/n_in + eps);
    for (int r = threadIdx.x; r < nrows_w; r += blockDim.x) {
        float v = 0.0f;
        for (int i = 0; i < nsplit; ++i) {
            v += __ldcg(part + i*(nrows_w + 1) + r);
        }
        const float y = v*scale + bias[r];
        mix[r] = y;
        if (r < n_sig) {
            sig[r] = hc_sigmoid(y);
        }
    }
    __syncthreads();

    if constexpr (vec4) {
        hc_pre_rms_norm_mul_row_vec4(pn.x, pn.pre, pn.w, pn.dst, pn.n, pn.hc, pn.scale_pre, pn.eps, pn.sx1, pn.sp0,
                s_pre, s_sum, pn.q8, pn.n_padded);
    } else {
        hc_pre_rms_norm_mul_row<HC_CHAIN_BLOCK, true>(pn.x, pn.pre, pn.w, pn.dst, pn.n, pn.hc, pn.scale_pre, pn.eps,
                pn.sx1, 0, pn.sp0, 0, 0, 0, s_pre, s_sum, pn.q8, pn.n_padded);
    }
}

void ggml_cuda_op_rms_norm_mat_vec_pre_norm(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul_mat, ggml_tensor * add, ggml_tensor * sig,
        const ggml_tensor * hc_pre, const ggml_tensor * rms_norm_pre, ggml_tensor * mul,
        void * dst_q8_1, const int64_t ncols_padded) {
    const ggml_tensor * x    = rms_norm->src[0];
    const ggml_tensor * w    = mul_mat->src[0];
    const ggml_tensor * bias = add->src[0] == mul_mat ? add->src[1] : add->src[0];
    const ggml_tensor * hx   = hc_pre->src[0];
    const ggml_tensor * pre  = hc_pre->src[1];
    const ggml_tensor * nw   = mul->src[0] == rms_norm_pre ? mul->src[1] : mul->src[0];

    const int     n       = x->ne[0];
    const int     nrows_w = w->ne[1];
    const int64_t sw1     = w->nb[1]/ggml_element_size(w);

    // same split as ggml_cuda_op_rms_norm_mat_vec_add for one token
    int nsplit = 16;
    while (nsplit > 1 && (n % (4*nsplit) != 0 || n/nsplit < 256)) {
        nsplit /= 2;
    }
    const int chunk = n/nsplit;

    const bool chain = ggml_nrows(x) == 1 && hx->ne[2] == 1 && chunk % 2 == 0 && sw1 % 2 == 0 &&
        ggml_is_contiguous(x) && ggml_is_contiguous(add) && ggml_is_contiguous(sig) && ggml_is_contiguous(mul) &&
        ggml_is_contiguous(nw) && hx->nb[0] == sizeof(float) && hx->ne[1] <= HC_MAX && sig->ne[0] <= nrows_w;
    if (!chain) {
        ggml_cuda_op_rms_norm_mat_vec_add(ctx, rms_norm, mul_mat, add, sig);
        ggml_cuda_op_hc_pre_rms_norm_mul(ctx, hc_pre, rms_norm_pre, mul, dst_q8_1, ncols_padded);
        return;
    }

    GGML_ASSERT(w->type == GGML_TYPE_F32 || w->type == GGML_TYPE_F16);
    GGML_ASSERT(bias->type == GGML_TYPE_F32 && pre->type == GGML_TYPE_F32 && nw->type == GGML_TYPE_F32);
    GGML_ASSERT(dst_q8_1 == nullptr || (ncols_padded >= hx->ne[0] && ncols_padded % QK8_1 == 0 && ncols_padded <= INT_MAX));

    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<float>        part(ctx.pool(), (size_t) nsplit*(nrows_w + 1));

    // a persistent counter keeps a memset node out of every CUDA graph launch;
    // it is created outside of stream capture, so a capture before the first eager run zeroes a temporary one
    unsigned int *                     counter = ctx.block_counters[ctx.curr_stream_no];
    ggml_cuda_pool_alloc<unsigned int> counter_tmp(ctx.pool());
    if (counter == nullptr) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
        if (capture == cudaStreamCaptureStatusNone) {
            CUDA_CHECK(cudaMalloc(&ctx.block_counters[ctx.curr_stream_no], sizeof(unsigned int)));
            counter = ctx.block_counters[ctx.curr_stream_no];
        } else {
            counter = counter_tmp.alloc(1);
        }
        CUDA_CHECK(cudaMemsetAsync(counter, 0, sizeof(unsigned int), stream));
    }

    hc_chain_pre_norm_params pn;
    pn.x         = (const float *) hx->data;
    pn.pre       = (const float *) pre->data;
    pn.w         = (const float *) nw->data;
    pn.dst       = (float *) mul->data;
    pn.n         = hx->ne[0];
    pn.hc        = hx->ne[1];
    pn.scale_pre = ggml_get_op_params_f32(hc_pre, 0);
    pn.eps       = ggml_get_op_params_f32(rms_norm_pre, 0);
    pn.sx1       = hx->nb[1]/sizeof(float);
    pn.sp0       = pre->nb[0]/sizeof(float);
    pn.q8        = (block_q8_1 *) dst_q8_1;
    pn.n_padded  = (int) ncols_padded;

    const float eps = ggml_get_op_params_f32(rms_norm, 0);
    // the pre-norm row with one float4 per thread: blockDim.x covers n (and the q8_1 padding), in whole warps
    const int  n_vec4  = (int) (dst_q8_1 ? ncols_padded : pn.n);
    const int  block4  = (n_vec4/4 + WARP_SIZE - 1)/WARP_SIZE*WARP_SIZE;
    const bool vec4    = pn.n % 4 == 0 && n_vec4 % 4 == 0 && block4 >= WARP_SIZE && block4 <= 1024 && pn.sx1 % 4 == 0 &&
        (uintptr_t) pn.x % 16 == 0 && (uintptr_t) pn.w % 16 == 0 && (uintptr_t) pn.dst % 16 == 0;
    const int  block   = vec4 ? block4 : HC_CHAIN_BLOCK;
    const dim3 grid(nsplit, (nrows_w + 1 + block/WARP_SIZE - 1)/(block/WARP_SIZE));
    const ggml_cuda_kernel_launch_params lp(grid, dim3(block), 0, stream);
    auto launch = [&](auto kernel, const auto * wd) {
        ggml_cuda_kernel_launch(kernel, lp, (const float *) x->data, wd, part.get(), counter, nrows_w, chunk, sw1,
                (const float *) bias->data, (float *) add->data, (float *) sig->data, (int) sig->ne[0], n, eps, pn);
    };
    if (w->type == GGML_TYPE_F16) {
        const half * wd = (const half *) w->data;
        vec4 ? launch(rms_norm_mat_vec_pre_norm_f32<half, true>, wd) : launch(rms_norm_mat_vec_pre_norm_f32<half, false>, wd);
    } else {
        const float * wd = (const float *) w->data;
        vec4 ? launch(rms_norm_mat_vec_pre_norm_f32<float, true>, wd) : launch(rms_norm_mat_vec_pre_norm_f32<float, false>, wd);
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// comb selector -> DSV4_HC_POST [-> stream sum]: one thread per (channel, token); every block computes the comb weights
// of its token: e = K^T mixes, rho = scale*max(e*r) + bias, comb(d, s) = e[d + hc*s]/rho + c0(d, s).
// All global loads of a thread are issued before the first barrier, so their latencies overlap; e is computed with
// one warp per output (the products of the mixes summed by warp_reduce_sum), the comb by the first warp

struct hc_select_kparams {
    const float * mixes;
    const float * K;
    const float * r;
    const float * c0;
    int     nmix;
    int64_t sm0, sm1;
    int64_t sk0, sk1;
    int64_t sr0;
    int64_t sc00, sc01;
    float   scale;
    float   bias;
};

template <bool with_sum>
static __global__ void hc_select_post_f32(
        const float * x, const float * res, const float * post, float * dst, float * sum,
        const hc_select_kparams sel, const int n, const int hc,
        const int64_t sx0, const int64_t sx1, const int64_t sr0, const int64_t sr1, const int64_t sr2,
        const int64_t sp0, const int64_t sp1,
        const int64_t sd0, const int64_t sd1, const int64_t sd2, const int64_t ss0, const int64_t ss1,
        block_q8_1 * q8, const int n_padded) {
    const int i       = blockIdx.x*blockDim.x + threadIdx.x;
    const int t       = blockIdx.y;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warp    = threadIdx.x / WARP_SIZE;
    const int n_warps = blockDim.x / WARP_SIZE;
    const int hc2     = hc*hc;

    __shared__ float s_post[HC_MAX];
    __shared__ float s_e[WARP_SIZE];
    __shared__ float s_comb[WARP_SIZE];

    ggml_cuda_pdl_sync();

    float r[HC_MAX];
    float xv = 0.0f;
#pragma unroll
    for (int s = 0; s < HC_MAX; ++s) {
        r[s] = s < hc && i < n ? res[i*sr0 + s*sr1 + t*sr2] : 0.0f;
    }
    if (i < n) {
        xv = x[i*sx0 + t*sx1];
    }

    float rr = 0.0f;
    float cc = 0.0f;
    if (warp == 0 && lane < hc2) {
        rr = sel.r[lane*sel.sr0];
        cc = sel.c0[(lane % hc)*sel.sc00 + (lane / hc)*sel.sc01];
    }
    if ((int) threadIdx.x < hc) {
        s_post[threadIdx.x] = post[threadIdx.x*sp0 + t*sp1];
    }

    for (int o = warp; o < hc2; o += n_warps) {
        float e = 0.0f;
        for (int k = lane; k < sel.nmix; k += WARP_SIZE) {
            e += sel.K[k*sel.sk0 + o*sel.sk1]*sel.mixes[k*sel.sm0 + t*sel.sm1];
        }
        e = warp_reduce_sum(e);
        if (lane == 0) {
            s_e[o] = e;
        }
    }
    __syncthreads();

    if (warp == 0) {
        const float e = lane < hc2 ? s_e[lane] : 0.0f;
        const float mx = warp_reduce_max(lane < hc2 ? e*rr : -FLT_MAX);
        if (lane < hc2) {
            const float rho = mx*sel.scale + sel.bias;
            s_comb[lane] = e/rho + cc;
        }
    }
    __syncthreads();

    float tot = 0.0f;
    if (i < n) {
        for (int d = 0; d < hc; ++d) {
            float v = xv*s_post[d];
#pragma unroll
            for (int s = 0; s < HC_MAX; ++s) {
                if (s < hc) {
                    v += r[s]*s_comb[d + hc*s];
                }
            }
            dst[i*sd0 + d*sd1 + t*sd2] = v;
            tot = d == 0 ? v : tot + v;
        }
        if constexpr (with_sum) {
            sum[i*ss0 + t*ss1] = tot;
        }
    }
    // n_padded is a multiple of WARP_SIZE: the condition is the same for all lanes of a warp
    if (with_sum && q8 != nullptr && i < n_padded) {
        quantize_q8_1_store(tot, q8, i);
    }
}

void ggml_cuda_op_hc_post_fused(ggml_backend_cuda_context & ctx, ggml_tensor * hc_post,
        const ggml_cuda_hc_select_args & sel, ggml_tensor * sum, void * sum_q8_1, const int64_t ncols_padded) {
    const ggml_tensor * x    = hc_post->src[0];
    const ggml_tensor * res  = hc_post->src[1];
    const ggml_tensor * post = hc_post->src[2];

    GGML_ASSERT(x->type == GGML_TYPE_F32 && res->type == GGML_TYPE_F32 && post->type == GGML_TYPE_F32);
    GGML_ASSERT(hc_post->type == GGML_TYPE_F32);
    GGML_ASSERT(sum == nullptr || sum->type == GGML_TYPE_F32);

    const int n  = x->ne[0];
    const int nt = x->ne[1];
    const int hc = res->ne[1];
    GGML_ASSERT(hc*hc <= WARP_SIZE);

    constexpr size_t f = sizeof(float);
    hc_select_kparams kp;
    kp.mixes = (const float *) sel.mixes->data;
    kp.K     = (const float *) sel.K->data;
    kp.r     = (const float *) sel.r->data;
    kp.c0    = (const float *) sel.c0->data;
    kp.nmix  = sel.K->ne[0];
    kp.sm0   = sel.mixes->nb[0]/f;
    kp.sm1   = sel.mixes->nb[1]/f;
    kp.sk0   = sel.K->nb[0]/f;
    kp.sk1   = sel.K->nb[1]/f;
    kp.sr0   = sel.r->nb[0]/f;
    kp.sc00  = sel.c0->nb[0]/f;
    kp.sc01  = sel.c0->nb[1]/f;
    kp.scale = sel.scale;
    kp.bias  = sel.bias;

    GGML_ASSERT(sum_q8_1 == nullptr || (sum != nullptr && nt == 1 && ncols_padded >= n && ncols_padded % QK8_1 == 0 &&
                ncols_padded <= INT_MAX));
    // with the q8_1 copy of the sum the threads cover its zero padding too
    const int n_threads = sum_q8_1 ? (int) ncols_padded : n;

    const int block_size = 256;
    const ggml_cuda_kernel_launch_params lp(dim3((n_threads + block_size - 1)/block_size, nt), dim3(block_size), 0, ctx.stream());
    ggml_cuda_kernel_launch(sum ? hc_select_post_f32<true> : hc_select_post_f32<false>, lp,
            (const float *) x->data, (const float *) res->data, (const float *) post->data,
            (float *) hc_post->data, sum ? (float *) sum->data : nullptr, kp, n, hc,
            (int64_t) (x->nb[0]/f), (int64_t) (x->nb[1]/f),
            (int64_t) (res->nb[0]/f), (int64_t) (res->nb[1]/f), (int64_t) (res->nb[2]/f),
            (int64_t) (post->nb[0]/f), (int64_t) (post->nb[1]/f),
            (int64_t) (hc_post->nb[0]/f), (int64_t) (hc_post->nb[1]/f), (int64_t) (hc_post->nb[2]/f),
            sum ? (int64_t) (sum->nb[0]/f) : (int64_t) 0, sum ? (int64_t) (sum->nb[1]/f) : (int64_t) 0,
            (block_q8_1 *) sum_q8_1, (int) ncols_padded);
}

// ---------------------------------------------------------------------------------------------------------------------
// RMS_NORM -> MUL -> RESHAPE -> ADD(streams, broadcast): one block per token. Every thread reads a stream element
// before it writes the same element, so dst may alias the streams

template <int block_size>
static __global__ void rms_norm_mul_add_streams_f32(
        const float * g, const float * w, const float * streams, float * dst, const int n, const int hc, const float eps,
        const int64_t sg1, const int64_t sx1, const int64_t sx2, const int64_t sd1, const int64_t sd2) {
    const int t = blockIdx.x;

    __shared__ float s_sum[32];

    ggml_cuda_pdl_sync();

    const float * gr = g + t*sg1;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += block_size) {
        ss += gr[i]*gr[i];
    }
    ss = block_reduce<block_reduce_method::SUM, block_size>(ss, s_sum);
    const float scale = rsqrtf(ss/n + eps);
    for (int i = threadIdx.x; i < n; i += block_size) {
        const float a = gr[i]*scale*w[i];
        for (int s = 0; s < hc; ++s) {
            dst[i + s*sd1 + t*sd2] = streams[i + s*sx1 + t*sx2] + a;
        }
    }
}

// the same for n % 4 == 0, hc <= HC_MAX and 16-byte aligned rows: one float4 of the row per thread (n/4 threads),
// the row, the weight and all streams of a thread are loaded once, before the reduction
static __global__ void rms_norm_mul_add_streams_f32_vec4(
        const float * g, const float * w, const float * streams, float * dst, const int n, const int hc, const float eps,
        const int64_t sg1, const int64_t sx1, const int64_t sx2, const int64_t sd1, const int64_t sd2) {
    const int t = blockIdx.x;
    const int i = 4*threadIdx.x;

    __shared__ float s_sum[32];

    ggml_cuda_pdl_sync();

    float4 gv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float4 wv = gv;
    float4 xv[HC_MAX];
#pragma unroll
    for (int s = 0; s < HC_MAX; ++s) {
        xv[s] = gv;
        if (s < hc && i < n) {
            xv[s] = *(const float4 *) (streams + s*sx1 + t*sx2 + i);
        }
    }
    if (i < n) {
        gv = *(const float4 *) (g + t*sg1 + i);
        wv = *(const float4 *) (w + i);
    }
    float ss = gv.x*gv.x;
    ss += gv.y*gv.y;
    ss += gv.z*gv.z;
    ss += gv.w*gv.w;
    ss = block_reduce<block_reduce_method::SUM>(ss, s_sum);
    const float scale = rsqrtf(ss/n + eps);
    if (i >= n) {
        return;
    }
    const float4 a = make_float4(gv.x*scale*wv.x, gv.y*scale*wv.y, gv.z*scale*wv.z, gv.w*scale*wv.w);
#pragma unroll
    for (int s = 0; s < HC_MAX; ++s) {
        if (s < hc) {
            *(float4 *) (dst + s*sd1 + t*sd2 + i) = make_float4(xv[s].x + a.x, xv[s].y + a.y, xv[s].z + a.z, xv[s].w + a.w);
        }
    }
}

void ggml_cuda_op_rms_norm_mul_add_streams(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm,
        const ggml_tensor * mul, ggml_tensor * add) {
    const ggml_tensor * g       = rms_norm->src[0];
    const ggml_tensor * w       = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const ggml_tensor * streams = add->src[0]->op == GGML_OP_RESHAPE && add->src[0]->src[0] == mul ? add->src[1] : add->src[0];

    GGML_ASSERT(g->type == GGML_TYPE_F32 && g->nb[0] == sizeof(float));
    GGML_ASSERT(w->type == GGML_TYPE_F32 && ggml_is_contiguous(w));
    GGML_ASSERT(streams->type == GGML_TYPE_F32 && streams->nb[0] == sizeof(float));
    GGML_ASSERT(add->type == GGML_TYPE_F32 && add->nb[0] == sizeof(float));

    constexpr size_t f = sizeof(float);
    const int     n   = add->ne[0];
    const int     hc  = add->ne[1];
    const int64_t sg1 = g->nb[1]/f;
    const int64_t sx1 = streams->nb[1]/f;
    const int64_t sx2 = streams->nb[2]/f;
    const int64_t sd1 = add->nb[1]/f;
    const int64_t sd2 = add->nb[2]/f;
    const float   eps = ggml_get_op_params_f32(rms_norm, 0);

    const int nth4 = (n/4 + WARP_SIZE - 1)/WARP_SIZE*WARP_SIZE;
    if (n > 0 && n % 4 == 0 && nth4 <= 1024 && hc <= HC_MAX && ggml_cuda_is_aligned(g, 16) && ggml_cuda_is_aligned(w, 16) &&
            ggml_cuda_is_aligned(streams, 16) && ggml_cuda_is_aligned(add, 16)) {
        const ggml_cuda_kernel_launch_params lp(dim3(add->ne[2]), dim3(nth4), 0, ctx.stream());
        ggml_cuda_kernel_launch(rms_norm_mul_add_streams_f32_vec4, lp,
                (const float *) g->data, (const float *) w->data, (const float *) streams->data, (float *) add->data,
                n, hc, eps, sg1, sx1, sx2, sd1, sd2);
        return;
    }

    constexpr int block_size = 256;
    const ggml_cuda_kernel_launch_params lp(dim3(add->ne[2]), dim3(block_size), 0, ctx.stream());
    ggml_cuda_kernel_launch(rms_norm_mul_add_streams_f32<block_size>, lp,
            (const float *) g->data, (const float *) w->data, (const float *) streams->data, (float *) add->data,
            n, hc, eps, sg1, sx1, sx2, sd1, sd2);
}
