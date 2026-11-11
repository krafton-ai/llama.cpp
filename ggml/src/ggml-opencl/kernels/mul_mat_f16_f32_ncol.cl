#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_khr_subgroup_shuffle
#pragma OPENCL EXTENSION cl_khr_subgroup_shuffle : enable
#elif defined(cl_qcom_subgroup_shuffle)
#pragma OPENCL EXTENSION cl_qcom_subgroup_shuffle : enable
#endif

// Decode attention of one KV head against NCOL query heads without flash attention, as two matrix
// multiplies with the heads as the columns:
//   scores[j][h] = sum_k K[j][k] * q[h][k]      kernel_mul_mat_f16_f32_ncol: K rows streamed once, q in local memory
#define NCOL 32

#define NCOL_RED8(v, o)                     \
    v.s0 += sub_group_shuffle_xor(v.s0, o); \
    v.s1 += sub_group_shuffle_xor(v.s1, o); \
    v.s2 += sub_group_shuffle_xor(v.s2, o); \
    v.s3 += sub_group_shuffle_xor(v.s3, o); \
    v.s4 += sub_group_shuffle_xor(v.s4, o); \
    v.s5 += sub_group_shuffle_xor(v.s5, o); \
    v.s6 += sub_group_shuffle_xor(v.s6, o); \
    v.s7 += sub_group_shuffle_xor(v.s7, o);

#define NCOL_STORE8(v, base, stride) \
    base[0 * stride] = v.s0;         \
    base[1 * stride] = v.s1;         \
    base[2 * stride] = v.s2;         \
    base[3 * stride] = v.s3;         \
    base[4 * stride] = v.s4;         \
    base[5 * stride] = v.s5;         \
    base[6 * stride] = v.s6;         \
    base[7 * stride] = v.s7;

//------------------------------------------------------------------------------
// D[n][m] = sum_k A[m][k] * B[n][k]; A f16 with contiguous rows of stride nba1 (8-byte aligned), B f32 rows of
// stride nbb1, n < NCOL, K % 4 == 0. Work-group 256 lanes: 8 lanes per row, 32 rows per pass, NCOL_ROWS_WG rows
// per group. B is staged in local memory as half, NCOL_KS k at a time, as Bl[k][n]: one k gives a lane the NCOL
// values with 4 half8 loads. Each lane holds the NCOL sums of its row over its half4 chunks of k; the 8 lanes of
// a row are summed with shuffles (so the 8 lanes of a row must share a subgroup: any subgroup size that is a
// multiple of 8).
//------------------------------------------------------------------------------
#define NCOL_LPR     8
#define NCOL_KS      256
#define NCOL_ROWS_WG 64

kernel void kernel_mul_mat_f16_f32_ncol(
        global const char * a,
        ulong offa,
        ulong nba1,
        global const char * b,
        ulong offb,
        ulong nbb1,
        global char * d,
        ulong offd,
        ulong nbd1,
        int K,
        int M) {
    local half Bl[NCOL_KS][NCOL];

    const int lid = get_local_id(0);  // 0..255
    const int rl = lid / NCOL_LPR;    // row within the pass, 0..31
    const int kl = lid % NCOL_LPR;    // lane within the row
    const int m0 = get_group_id(0) * NCOL_ROWS_WG;

    a += offa;
    b += offb;
    d += offd;

    for (int p = 0; p < NCOL_ROWS_WG / 32; ++p) {
        const int m = m0 + p * 32 + rl;
        global const half * arow = (global const half *) (a + (ulong) min(m, M - 1) * nba1);
        float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;

        for (int k0 = 0; k0 < K; k0 += NCOL_KS) {
            {  // stage B[0..NCOL)[k0..k0+KS): lane -> column lid/8, run of 32 k at (lid%8)*32
                const int n = lid / 8;
                const int ks = (lid % 8) * 32;
                global const float * bn = (global const float *) (b + (ulong) n * nbb1) + k0 + ks;
                for (int i = 0; i < 32; ++i) {
                    Bl[ks + i][n] = (k0 + ks + i < K) ? (half) bn[i] : (half) 0.0f;
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);

            const int kend = min(K - k0, NCOL_KS);
            for (int c = kl * 4; c < kend; c += NCOL_LPR * 4) {
                const float4 av = convert_float4(vload4(0, arow + k0 + c));
#define NCOL_MAD_K(ai, kk)                                       \
    {                                                            \
        const float8 b0 = convert_float8(vload8(0, &Bl[kk][0])); \
        const float8 b1 = convert_float8(vload8(1, &Bl[kk][0])); \
        const float8 b2 = convert_float8(vload8(2, &Bl[kk][0])); \
        const float8 b3 = convert_float8(vload8(3, &Bl[kk][0])); \
        const float8 a8 = (float8) (ai);                         \
        acc0 = mad(a8, b0, acc0);                                \
        acc1 = mad(a8, b1, acc1);                                \
        acc2 = mad(a8, b2, acc2);                                \
        acc3 = mad(a8, b3, acc3);                                \
    }
                NCOL_MAD_K(av.s0, c + 0)
                NCOL_MAD_K(av.s1, c + 1)
                NCOL_MAD_K(av.s2, c + 2)
                NCOL_MAD_K(av.s3, c + 3)
#undef NCOL_MAD_K
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        NCOL_RED8(acc0, 1u)
        NCOL_RED8(acc1, 1u)
        NCOL_RED8(acc2, 1u) NCOL_RED8(acc3, 1u) NCOL_RED8(acc0, 2u) NCOL_RED8(acc1, 2u) NCOL_RED8(acc2, 2u)
                NCOL_RED8(acc3, 2u) NCOL_RED8(acc0, 4u) NCOL_RED8(acc1, 4u) NCOL_RED8(acc2, 4u) NCOL_RED8(acc3, 4u)

                        if (kl == 0 && m < M) {
            global float * out = (global float *) (d + (ulong) m * sizeof(float));
            const int s = (int) (nbd1 / sizeof(float));
            NCOL_STORE8(acc0, out, s) out += 8 * s;
            NCOL_STORE8(acc1, out, s) out += 8 * s;
            NCOL_STORE8(acc2, out, s) out += 8 * s;
            NCOL_STORE8(acc3, out, s)
        }
    }
}
