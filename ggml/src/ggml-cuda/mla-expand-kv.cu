#include "mla-expand-kv.cuh"
#include "convert.cuh"

// mla_expand_kv: K = [W_UK latent | RoPE part] and V = W_UV latent of heads h0 .. h0 + g - 1, in F16 straight from the
// F16 latent cache. Two strided batched GEMMs (one batch per head, F32 accumulation, F16 output written in place into
// K [nk, g, n_kv] and V [nv, g, n_kv] through ldc) and a copy of the shared RoPE part into every head's K row

static __global__ void mla_expand_kv_rope(const half * kv, half * K, const int64_t ldk, const int rank, const int nr,
        const int nn, const int nk, const int g, const int64_t n_kv) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_kv*g*nr) {
        return;
    }
    const int     j  = i % nr;
    const int64_t th = i / nr;          // t*g + h
    const int64_t t  = th / g;
    K[th*nk + nn + j] = kv[t*ldk + rank + j];
}

// the same with 16-byte copies (8 halves) when every offset is a multiple of 8 halves
static __global__ void mla_expand_kv_rope_v8(const int4 * kv, int4 * K, const int64_t ldk8, const int rank8, const int nr8,
        const int nn8, const int nk8, const int g, const int64_t n_kv) {
    const int64_t i = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n_kv*g*nr8) {
        return;
    }
    const int     j  = i % nr8;
    const int64_t th = i / nr8;
    const int64_t t  = th / g;
    K[th*nk8 + nn8 + j] = kv[t*ldk8 + rank8 + j];
}

void ggml_cuda_op_mla_expand_kv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kv  = dst->src[0];
    const ggml_tensor * k_b = dst->src[1];
    const ggml_tensor * v_b = dst->src[2];
    const int h0   = ggml_get_op_params_i32(dst, 0);
    const int g    = ggml_get_op_params_i32(dst, 1);
    const int rank = (int) k_b->ne[1];
    const int     nn = (int) k_b->ne[0], nr = (int) (kv->ne[0] - rank), nk = nn + nr, nv = (int) v_b->ne[1];
    const int64_t n_kv = kv->ne[1];
    const int64_t ldk  = kv->nb[1]/sizeof(half);
    GGML_ASSERT(kv->nb[1] % sizeof(half) == 0);

    half * K = (half *) dst->data;
    half * V = K + (int64_t) nk*g*n_kv;
    cudaStream_t stream = ctx.stream();

    // the weights of heads h0 .. h0 + g - 1 in F16 (dequantized into the pool unless already F16)
    ggml_cuda_pool_alloc<half> wk_f16(ctx.pool()), wv_f16(ctx.pool());
    auto heads_f16 = [&](const ggml_tensor * w, int64_t n_per_head, ggml_cuda_pool_alloc<half> & buf) -> const half * {
        const int64_t off = (int64_t) h0*n_per_head, n = n_per_head*g;
        if (w->type == GGML_TYPE_F16) {
            return (const half *) w->data + off;
        }
        const int64_t blk = ggml_blck_size(w->type);
        GGML_ASSERT(off % blk == 0 && n % blk == 0);
        buf.alloc(n);
        ggml_get_to_fp16_cuda(w->type)((const char *) w->data + off/blk*ggml_type_size(w->type), buf.get(), n, stream);
        return buf.get();
    };
    const half * wkh = heads_f16(k_b, (int64_t) nn*rank, wk_f16);
    const half * wvh = heads_f16(v_b, (int64_t) rank*nv, wv_f16);

    cublasHandle_t handle = ctx.cublas_handle();
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    const float alpha = 1.0f, beta = 0.0f;
    // column-major, C (m x n_kv) per head: K = W_UK (nn x rank, lda nn, as stored) * latent; V = W_UV^T (W_UV rows of rank)
    // * latent; B = latent columns of the cache rows (ldb = cache row stride); C written in place through ldc
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, nn, (int) n_kv, rank,
            &alpha, wkh, CUDA_R_16F, nn, (int64_t) nn*rank,
                    (const half *) kv->data, CUDA_R_16F, (int) ldk, 0,
            &beta,  K, CUDA_R_16F, nk*g, nk, g, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, nv, (int) n_kv, rank,
            &alpha, wvh, CUDA_R_16F, rank, (int64_t) nv*rank,
                    (const half *) kv->data, CUDA_R_16F, (int) ldk, 0,
            &beta,  V, CUDA_R_16F, nv*g, nv, g, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    if (nr > 0) {
        if (ldk % 8 == 0 && rank % 8 == 0 && nr % 8 == 0 && nn % 8 == 0 && nk % 8 == 0 &&
                (uintptr_t) kv->data % 16 == 0 && (uintptr_t) K % 16 == 0) {
            const int64_t n = n_kv*g*(nr/8);
            mla_expand_kv_rope_v8<<<(n + 255)/256, 256, 0, stream>>>(
                (const int4 *) kv->data, (int4 *) K, ldk/8, rank/8, nr/8, nn/8, nk/8, g, n_kv);
        } else {
            const int64_t n = n_kv*g*nr;
            mla_expand_kv_rope<<<(n + 255)/256, 256, 0, stream>>>((const half *) kv->data, K, ldk, rank, nr, nn, nk, g, n_kv);
        }
        CUDA_CHECK(cudaGetLastError());
    }
}
