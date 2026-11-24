#include "common.h"
#include "dequantize.h"

// dequantize a quantized KV cache tensor to contiguous F16 before running the F16 flash attention kernels
// - one thread per block; dispatched separately for K and V
// - ref: https://github.com/ggml-org/llama.cpp/pull/27390
template <
    typename block_t,
    short QK,
    void (*deq_t4x4)(device const block_t *, short, thread float4x4 &)>
kernel void kernel_flash_attn_ext_kv_f16(
        constant ggml_metal_kargs_flash_attn_ext_kv_f16 & args,
        device const char * x,
        device       half * x_dst,
        uint gid [[thread_position_in_grid]]) {
    if (gid >= (uint) args.nblocks) {
        return;
    }

    const uint nb = args.ne0/QK;
    const uint i0 = gid%nb;
    uint ib       = gid/nb;
    const uint i1 = ib%args.ne1;
    ib /= args.ne1;
    const uint i2 = ib%args.ne2;
    const uint i3 = ib/args.ne2;

    const uint64_t offs = i0*args.nb0 + i1*args.nb1 + i2*args.nb2 + i3*args.nb3;

    device const block_t * src = (device const block_t *) (x + offs);
    device half4 * dst = (device half4 *) x_dst + (QK/4)*gid;

    for (short i = 0; i < QK/16; ++i) {
        float4x4 reg;
        deq_t4x4(src, i, reg);
        dst[4*i + 0] = (half4) reg[0];
        dst[4*i + 1] = (half4) reg[1];
        dst[4*i + 2] = (half4) reg[2];
        dst[4*i + 3] = (half4) reg[3];
    }
}

typedef decltype(kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>) kernel_flash_attn_ext_kv_f16_t;

template [[host_name("kernel_flash_attn_ext_kv_q4_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_0, 32, dequantize_q4_0>;
template [[host_name("kernel_flash_attn_ext_kv_q4_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q4_1, 32, dequantize_q4_1>;
template [[host_name("kernel_flash_attn_ext_kv_q5_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_0, 32, dequantize_q5_0>;
template [[host_name("kernel_flash_attn_ext_kv_q5_1_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q5_1, 32, dequantize_q5_1>;
template [[host_name("kernel_flash_attn_ext_kv_q8_0_f16")]] kernel kernel_flash_attn_ext_kv_f16_t kernel_flash_attn_ext_kv_f16<block_q8_0, 32, dequantize_q8_0>;

constant bool FC_flash_attn_ext_pad_has_mask [[function_constant(FC_FLASH_ATTN_EXT_PAD + 0)]];

constant int32_t FC_flash_attn_ext_pad_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_PAD + 25)]];

// pad the last chunk of C elements of k and v into a an extra pad buffer
kernel void kernel_flash_attn_ext_pad(
        constant ggml_metal_kargs_flash_attn_ext_pad & args,
        device const char * k,
        device const char * v,
        device const char * mask,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t C = FC_flash_attn_ext_pad_ncpsg;

    device char * k_pad    = dst;
    device char * v_pad    = k_pad + args.nb11*C*args.ne_12_2*args.ne_12_3;
    device char * mask_pad = v_pad + args.nb21*C*args.ne_12_2*args.ne_12_3;

    const int32_t icp = args.ne11 % C;
    const int32_t ic0 = args.ne11 - icp;

    const int32_t i1 = tgpig[0];
    const int32_t i2 = tgpig[1];
    const int32_t i3 = tgpig[2];

    if (i2 < args.ne_12_2 && i3 < args.ne_12_3) {
        device const char * k_src = k + args.nb11*(ic0 + i1) + args.nb12*i2 + args.nb13*i3;
        device const char * v_src = v + args.nb21*(ic0 + i1) + args.nb22*i2 + args.nb23*i3;

        device char * k_dst = k_pad + args.nb11*i1 + args.nb11*C*i2 + args.nb11*C*args.ne_12_2*i3;
        device char * v_dst = v_pad + args.nb21*i1 + args.nb21*C*i2 + args.nb21*C*args.ne_12_2*i3;

        if (i1 >= icp) {
            // here it is not important the exact value that will be used as we rely on masking out the scores in the attention
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = 0;
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = 0;
            }
        } else {
            for (uint64_t i = tiitg; i < args.nb11; i += ntg.x) {
                k_dst[i] = k_src[i];
            }
            for (uint64_t i = tiitg; i < args.nb21; i += ntg.x) {
                v_dst[i] = v_src[i];
            }
        }
    }

    if (FC_flash_attn_ext_pad_has_mask) {
        if (i2 < args.ne32 && i3 < args.ne33) {
            for (int ib = i1; ib < args.ne31; ib += C) {
                device const half * mask_src = (device const half *)(mask      + args.nb31*ib + args.nb32*i2 + args.nb33*i3) + ic0;
                device       half * mask_dst = (device       half *)(mask_pad) + C*ib + C*args.ne31*i2 + C*args.ne31*args.ne32*i3;

                for (int i = tiitg; i < C; i += ntg.x) {
                    if (i >= icp) {
                        mask_dst[i] = -MAXHALF;
                    } else {
                        mask_dst[i] = mask_src[i];
                    }
                }
            }
        }
    }
}

constant int32_t FC_flash_attn_ext_blk_nqptg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 24)]];
constant int32_t FC_flash_attn_ext_blk_ncpsg [[function_constant(FC_FLASH_ATTN_EXT_BLK + 25)]];

// scan the blocks of the mask that are not masked
// 0 -     masked (i.e. full of -INF, skip)
// 1 - not masked (i.e. at least one element of the mask is not -INF)
// 2 - all zero
kernel void kernel_flash_attn_ext_blk(
        constant ggml_metal_kargs_flash_attn_ext_blk & args,
        device const char * mask,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    // block size C x Q
    const int32_t Q = FC_flash_attn_ext_blk_nqptg;
    const int32_t C = FC_flash_attn_ext_blk_ncpsg;

    constexpr short NW  = N_SIMDWIDTH;

    const int32_t i3 = tgpig[2]/args.ne32;
    const int32_t i2 = tgpig[2]%args.ne32;
    const int32_t i1 = tgpig[1];
    const int32_t i0 = tgpig[0];

    char res = i0*C + C > args.ne30 || i1*Q + Q > args.ne31 ? 1 : 0;

    device const half * mask_src = (device const half *) (mask + (i1*Q)*args.nb31 + i2*args.nb32 + i3*args.nb33) + i0*C + tiisg;

    // detailed check of the elements of the block
    if ((C > NW || Q > 1) && res == 0) {
        half mmin =  MAXHALF;
        half mmax = -MAXHALF;

        FOR_UNROLL (short j = 0; j < Q; ++j) {
            FOR_UNROLL (short ii = 0; ii < C/NW; ++ii) {
                mmin = min(mmin, mask_src[ii*NW]);
                mmax = max(mmax, mask_src[ii*NW]);
            }

            mask_src += args.nb31/2;
        }

        mmin = simd_min(mmin);
        mmax = simd_max(mmax);

        if (mmax > -MAXHALF) {
            if (mmin == 0.0 && mmax == 0.0) {
                res = 2;
            } else {
                res = 1;
            }
        }
    }

    const int32_t nblk1 = ((args.ne01 + Q - 1)/Q);
    const int32_t nblk0 = ((args.ne30 + C - 1)/C);

    if (tiisg == 0) {
        dst[((i3*args.ne32 + i2)*nblk1 + i1)*nblk0 + i0] = res;
    }
}
// compress the finite entries of each KQ mask row into a list of KV indices (ascending order),
// padded with -1 up to n_kv_max_padded (a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG)
// one threadgroup per mask row; the mask remains the single source of truth for the values
kernel void kernel_flash_attn_ext_vec_idx(
        constant ggml_metal_kargs_flash_attn_ext_vec_idx & args,
        device const half * mask,
        device       int  * idx,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiitg[[thread_index_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    constexpr short NW = N_SIMDWIDTH;
    constexpr short NLOCAL = 32; // max finite positions kept in registers per thread

    const int i1 = tgpig[0];
    const int i2 = tgpig[1];
    const int i3 = tgpig[2];

    device const half * pm  = (device const half *) ((device const char *) mask + i1*args.nb31 + i2*args.nb32 + i3*args.nb33);
    device int * pidx = idx + (((int64_t)i3*args.ne32 + i2)*args.ne31 + i1)*args.n_kv_max_padded;

    const int n  = args.ne30;
    const int q  = n/ntg.x;
    const int r  = n%ntg.x;

    // each thread handles a contiguous slice of the mask row
    const int r0 = q*tiitg + min((int) tiitg, r);
    const int r1 = r0 + q + (tiitg < r ? 1 : 0);

    // count the finite entries in the slice and keep their positions in registers (single mask read)
    int cnt = 0;  // total finite entries in the slice
    int nloc = 0; // finite entries kept in registers
    int local[NLOCAL];
    for (int i = r0; i < r1; ++i) {
        if (isfinite((float) pm[i])) {
            if (nloc < NLOCAL) {
                local[nloc] = i;
                nloc++;
            }
            cnt++;
        }
    }

    const short sgitg = tiitg/NW;
    const short tiisg = tiitg%NW;

    threadgroup int tcount[8];

    // simd_sum is a collective: all lanes must evaluate it
    const int sg_sum = simd_sum(cnt);
    if (tiisg == 0) {
        tcount[sgitg] = sg_sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    int total = 0;
    for (short s = 0; s < ntg.x/NW; ++s) {
        total += tcount[s];
    }

    // base offset of this thread's slice in the output list (exclusive scan within the simdgroup)
    int sg_base = 0;
    for (short s = 0; s < sgitg; ++s) {
        sg_base += tcount[s];
    }

    // exclusive prefix scan of the per-thread counts within the simdgroup
    int incl = cnt;
    for (int d = 1; d < NW; d <<= 1) {
        const int v = simd_shuffle_up(incl, d);
        if (tiisg >= d) {
            incl += v;
        }
    }
    const int base = sg_base + (incl - cnt);

    // write the finite positions in order; if the hint is violated, keep only the first n_kv_max entries
    int j = 0;
    for (; j < nloc && base + j < args.n_kv_max; ++j) {
        pidx[base + j] = local[j];
    }

    // a dense mask may have more than NLOCAL finite entries in a slice; re-read the mask to write the rest
    if (cnt > nloc && base + nloc < args.n_kv_max) {
        int j2 = 0;
        for (int i = r0; i < r1; ++i) {
            if (isfinite((float) pm[i])) {
                if (j2 >= nloc) {
                    pidx[base + j2] = i;
                }
                j2++;
                if (base + j2 >= args.n_kv_max) {
                    break;
                }
            }
        }
    }

    // pad the tail of the list with -1
    const int count = min(total, args.n_kv_max);
    for (int i = count + tiitg; i < args.n_kv_max_padded; i += ntg.x) {
        pidx[i] = -1;
    }
}

constant int32_t FC_flash_attn_ext_vec_reduce_DV  [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 0)]];
constant int32_t FC_flash_attn_ext_vec_reduce_NWG [[function_constant(FC_FLASH_ATTN_EXT_VEC_REDUCE + 1)]];

kernel void kernel_flash_attn_ext_vec_reduce(
        constant ggml_metal_kargs_flash_attn_ext_vec_reduce & args,
        device  const char * htmp,
        device        char * dst,
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#define NWG (FC_flash_attn_ext_vec_reduce_NWG)
#define DV  (FC_flash_attn_ext_vec_reduce_DV)

    const uint64_t rid = tgpig;

    const short iwg = tiisg;

    device const float  * ss    = (device const float  *) htmp + (uint64_t)args.nrows*DV*NWG;

    float S = ss[rid*(2*NWG) + 2*iwg + 0];
    float M = ss[rid*(2*NWG) + 2*iwg + 1];

    const float m  = simd_max(M);
    const float ms = exp(M - m);

    S = simd_sum(S*ms);
    S = S == 0.0f ? 0.0f : 1.0f/S;

    const short DV4 = DV/4;

    device const float4 * htmp4 = (device const float4 *) htmp + rid*DV4*NWG;
    device       float4 * dst4  = (device       float4 *) dst  + rid*DV4;

    for (short i = sgitg; i < DV4; i += NWG) {
        const float4 v = simd_sum(htmp4[i*NWG + iwg]*ms);

        if (iwg == 0) {
            dst4[i] = v*S;
        }
    }

#undef NWG
#undef DV
}

constant short FC_lightning_indexer_nh [[function_constant(FC_LIGHTNING_INDEXER + 0)]];

template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK    = OP_LIGHTNING_INDEXER_DK;
    const     short NH    = FC_lightning_indexer_nh;
    constexpr short NHPTG = OP_LIGHTNING_INDEXER_NHPTG;
    constexpr short NKPSG = OP_LIGHTNING_INDEXER_NKPSG;
    constexpr short NSG   = OP_LIGHTNING_INDEXER_NSG;
    constexpr short NBPTG = OP_LIGHTNING_INDEXER_NBPTG;

    constexpr short DK4  = DK/4;
    constexpr short DK8  = DK/8;
    constexpr short DK16 = DK/16;

    constexpr short NK  = NKPSG*NSG; // keys    per threadgroup
    constexpr short NTG = 32*NSG;    // threads per threadgroup

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;            // first key of this threadgroup
    const int i_kv     = i_kv_0 + sgitg*NKPSG;  // first key of this simdgroup

    threadgroup half sk[NK * DK16 * 16];
    threadgroup half4x4 * sk4x4 = (threadgroup half4x4 *) sk;

    for (short i = tiitg; i < NK*DK16; i += NTG) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;

        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (i_kv_0 + ik)*args.nbk2 + i_stream*args.nbk3);

            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }

        sk4x4[i] = tmp;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // K tile of this simdgroup, transposed to [DK, NKPSG]
    simdgroup_half8x8 mk[DK8];

    FOR_UNROLL (short i = 0; i < DK8; ++i) {
        simdgroup_load(mk[i], sk + sgitg*NKPSG*DK + 8*i, DK, 0, true);
    }

    threadgroup half4   sq4[NHPTG*DK4];
    threadgroup half  * sq = (threadgroup half *) sq4;

    threadgroup float sw [NHPTG];
    threadgroup float sqk[NSG*NHPTG*NKPSG];

    const int i_batch_0 = tgpig.y*NBPTG;
    const int n_batch   = min((int) NBPTG, args.n_batch - i_batch_0);

    for (short ib = 0; ib < n_batch; ++ib) {
        const int i_batch = i_batch_0 + ib;

        device const char * pq = q + i_batch*args.nbq2 + i_stream*args.nbq3;
        device const char * pw = w + i_batch*args.nbw1 + i_stream*args.nbw3;

        float score = 0.0f;

        FOR_UNROLL (short i_head = 0; i_head < NH; i_head += NHPTG) {
            // stage the Q tile [DK, NHPTG] and the (prescaled) head weights, heads past NH are zero
            for (short i = tiitg; i < NHPTG*DK4; i += NTG) {
                const short ih = i/DK4;
                const short i4 = i%DK4;

                if (i_head + ih < NH) {
                    device const float4 * q4 = (device const float4 *) (pq + (i_head + ih)*args.nbq1);

                    sq4[ih*DK4 + i4] = half4(q4[i4]);
                } else {
                    sq4[ih*DK4 + i4] = half4(0.0h);
                }
            }

            if (tiitg < NHPTG) {
                sw[tiitg] = i_head + tiitg < NH ? ((device const float *) pw)[i_head + tiitg] : 0.0f;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);

            FOR_UNROLL (short i = 0; i < DK8; ++i) {
                simdgroup_half8x8 mq;

                simdgroup_load(mq, sq + 8*i, DK, 0, false);
                simdgroup_multiply_accumulate(mqk, mq, mk[i], mqk);
            }

            threadgroup float * pqk = sqk + sgitg*NHPTG*NKPSG;

            simdgroup_store(mqk, pqk, NKPSG, 0, false);
            simdgroup_barrier(mem_flags::mem_threadgroup);

            // one lane per key: ReLU, apply the head weight and accumulate over the head tile
            if (tiisg < NKPSG) {
                FOR_UNROLL (short ih = 0; ih < NHPTG; ++ih) {
                    score += max(pqk[ih*NKPSG + tiisg], 0.0f)*sw[ih];
                }
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiisg < NKPSG) {
            const int ik = i_kv + tiisg;
            if (ik < args.n_kv) {
                device const half  * pm = (device const half  *) (m   + i_batch*args.nbm1 + (i_stream % args.mask_ne3)*args.nbm3);
                device       float * pd = (device       float *) (dst + i_batch*args.nb1  + i_stream*args.nb3);

                pd[ik] = score + (float) pm[ik];
            }
        }
    }
}

typedef decltype(kernel_lightning_indexer<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_t;

template [[host_name("kernel_lightning_indexer_f32")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<float4x4, 1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_f16")]]  kernel kernel_lightning_indexer_t kernel_lightning_indexer<half4x4,  1, dequantize_f16>;

#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_lightning_indexer_bf16")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<bfloat4x4, 1, dequantize_bf16>;
#endif

template [[host_name("kernel_lightning_indexer_q4_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_q4_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_q5_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_q5_1")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_q8_0")]] kernel kernel_lightning_indexer_t kernel_lightning_indexer<block_q8_0, 2, dequantize_q8_0>;

// decode attention for the MLA latent MQA (dk576/dv512, one F16 KV head, V = first 512 elements of the K row).
// One threadgroup = 8 heads of one query token x one KV split (NWG splits, each a contiguous range of 64-key chunks);
// S = Q*K^T and O += P*V use 8x8 simdgroup MMAs; the per-split partial (unnormalized O, S, M) goes to the same tmp layout
// the vec kernel uses, so kernel_flash_attn_ext_vec_reduce combines the splits.
kernel void kernel_flash_attn_ext_mla_dec(
        constant ggml_metal_kargs_flash_attn_ext_mla_dec & args,
        device const char * q,
        device const char * k,
        device const char * mask,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]]) {
    constexpr short DK = 576;
    constexpr short DV = 512;
    constexpr short C  = 64;

    const int hb  = tgpig.x;
    const int iq1 = tgpig.y;
    const int iwg = tgpig.z;
    const int t   = sgitg*32 + tiisg;

    threadgroup half  * sq  = (threadgroup half  *) (shmem);            // [8][576]
    threadgroup float * sS  = (threadgroup float *) (shmem + 9216);     // [8][64]
    threadgroup half  * sP  = (threadgroup half  *) (shmem + 11264);    // [8][64]
    threadgroup float * sMS = (threadgroup float *) (shmem + 12288);    // [8]
    threadgroup float * sD  = (threadgroup float *) (shmem + 12352);    // [4][64]
    threadgroup float * sO  = (threadgroup float *) (shmem);            // [8][512] (aliases sq/sS/sP: only used after the KV loop)

    // queries: 8 heads of token iq1
    for (int i = t; i < 8*(DK/4); i += 128) {
        const int r = i/(DK/4);
        const int c = i%(DK/4);
        device const float4 * qq = (device const float4 *) (q + iq1*args.nb01 + (hb*8 + r)*args.nb02);
        ((threadgroup half4 *) (sq + r*DK))[c] = (half4) qq[c];
    }

    const int nch = args.ne11/C;
    const int c0  = (iwg*nch)/args.nwg;
    const int c1  = ((iwg + 1)*nch)/args.nwg;

    const int r  = t/16;
    const int cb = (t%16)*4;

    float M     = -FLT_MAX/2;
    float Spart = 0.0f;

    simdgroup_float8x8 mO[16];
    for (short j = 0; j < 16; ++j) {
        mO[j] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    const int ns10 = (int) (args.nb11/2);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const char * mrow = mask + iq1*args.nb31;

    for (int c = c0; c < c1; ++c) {
        const int ic = c*C;

        // S = Q*K^T for key blocks 2*sgitg, 2*sgitg + 1
        {
            device const half * pk0 = (device const half *) (k + (ic + (2*sgitg + 0)*8)*args.nb11);
            device const half * pk1 = (device const half *) (k + (ic + (2*sgitg + 1)*8)*args.nb11);

            simdgroup_float8x8 a0 = make_filled_simdgroup_matrix<float, 8>(0.0f);
            simdgroup_float8x8 a1 = make_filled_simdgroup_matrix<float, 8>(0.0f);

            simdgroup_half8x8 mq;
            simdgroup_half8x8 mk0;
            simdgroup_half8x8 mk1;

            for (short i = 0; i < DK/8; ++i) {
                simdgroup_load(mk0, pk0 + 8*i, ns10, 0, true);
                simdgroup_load(mk1, pk1 + 8*i, ns10, 0, true);
                simdgroup_load(mq,  sq  + 8*i, DK);

                simdgroup_multiply_accumulate(a0, mq, mk0, a0);
                simdgroup_multiply_accumulate(a1, mq, mk1, a1);
            }

            simdgroup_store(a0, sS + (2*sgitg + 0)*8, C);
            simdgroup_store(a1, sS + (2*sgitg + 1)*8, C);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax: 16 threads per row, 4 columns each
        {
            const float4 s4 = *((threadgroup float4 *) (sS + r*C + cb));
            const half4  m4 = *((device const half4 *) (mrow + (ic + cb)*2));
            const float4 s  = s4*args.scale + float4(m4);

            float mx = max(max(s[0], s[1]), max(s[2], s[3]));
            mx = max(mx, simd_shuffle_xor(mx, 1));
            mx = max(mx, simd_shuffle_xor(mx, 2));
            mx = max(mx, simd_shuffle_xor(mx, 4));
            mx = max(mx, simd_shuffle_xor(mx, 8));

            const float Mn = max(M, mx);
            const float ms = exp(M - Mn);
            const float4 p = exp(s - Mn);

            Spart = Spart*ms + (p[0] + p[1] + p[2] + p[3]);
            M = Mn;

            *((threadgroup half4 *) (sP + r*C + cb)) = half4(p);

            if ((t%16) == 0) {
                sMS[r] = ms;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = diag(ms)*O
        {
            for (short e = tiisg; e < 64; e += 32) {
                sD[sgitg*64 + e] = (e/8 == e%8) ? sMS[e/8] : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);

            simdgroup_float8x8 mD;
            simdgroup_load(mD, sD + sgitg*64, 8);

            for (short j = 0; j < 16; ++j) {
                simdgroup_float8x8 tmp;
                simdgroup_multiply(tmp, mD, mO[j]);
                mO[j] = tmp;
            }
        }

        // O += P*V (V = first 512 elements of the K row), this simdgroup owns 128 output columns
        for (short kb = 0; kb < C/8; ++kb) {
            simdgroup_half8x8 mP;
            simdgroup_load(mP, sP + kb*8, C);

            device const half * pv = (device const half *) (k + (ic + kb*8)*args.nb11) + sgitg*128;

            for (short j = 0; j < 16; ++j) {
                simdgroup_half8x8 mv;
                simdgroup_load(mv, pv + 8*j, ns10);
                simdgroup_multiply_accumulate(mO[j], mP, mv, mO[j]);
            }
        }
    }

    // total S of the row
    float Sx = Spart;
    Sx += simd_shuffle_xor(Sx, 1);
    Sx += simd_shuffle_xor(Sx, 2);
    Sx += simd_shuffle_xor(Sx, 4);
    Sx += simd_shuffle_xor(Sx, 8);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short j = 0; j < 16; ++j) {
        simdgroup_store(mO[j], sO + sgitg*128 + j*8, DV);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int64_t nrows = (int64_t) args.ne1*args.ne2;

    device float4 * dst4 = (device float4 *) dst;
    device float  * dst1 = (device float  *) dst + nrows*DV*args.nwg;

    for (int i = t; i < 8*(DV/4); i += 128) {
        const int rr = i/(DV/4);
        const int cc = i%(DV/4);
        const int64_t rid = (int64_t) (hb*8 + rr) + (int64_t) iq1*args.ne1;
        dst4[rid*(DV/4)*args.nwg + (int64_t) args.nwg*cc + iwg] = ((threadgroup float4 *) (sO + rr*DV))[cc];
    }

    if ((t%16) == 0) {
        const int64_t rid = (int64_t) (hb*8 + r) + (int64_t) iq1*args.ne1;
        dst1[rid*(2*args.nwg) + 2*iwg + 0] = Sx;
        dst1[rid*(2*args.nwg) + 2*iwg + 1] = M;
    }
}
