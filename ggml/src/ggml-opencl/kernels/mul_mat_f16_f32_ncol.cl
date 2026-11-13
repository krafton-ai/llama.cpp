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
//   out[d][h]    = sum_j V[j][d] * P[h][j]      kernel_mul_mat_f16t_f32_ncol_split (+ _reduce): V read in its cache
//                                               layout, so the transposed copy the graph asks for is not made
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

//------------------------------------------------------------------------------
// part[s][n][d] = sum_{j in split s} V[j][d] * P[n][j]; V f16 with d contiguous and row stride nbvj (the latent
// cache rows), P f32 rows of stride nbp1. Work-group 64 lanes = 64 consecutive d (one coalesced row read per j),
// one group per (d block, split of JC rows); the split's P rows sit in local memory and every lane reads the
// same NCOL values of a j (broadcast). kernel_mul_mat_ncol_split_reduce sums the splits into D[n][d].
//------------------------------------------------------------------------------
#define NCOL_JC 128

kernel void kernel_mul_mat_f16t_f32_ncol_split(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D,
        int JC) {
    local float Pl[NCOL_JC][NCOL + 4];

    const int lid = get_local_id(0);  // 0..63
    const int d = get_group_id(0) * 64 + lid;
    const int s = get_group_id(1);
    const int j0 = s * JC;
    const int jn = min(JC, J - j0);

    v += offv;
    p += offp;

    for (int n = 0; n < NCOL; ++n) {
        global const float * pn = (global const float *) (p + (ulong) n * nbp1) + j0;
        for (int jj = lid; jj < NCOL_JC; jj += 64) {
            Pl[jj][n] = jj < jn ? pn[jj] : 0.0f;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (d >= D) {
        return;
    }
    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    global const char * vd = v + (ulong) j0 * nbvj + (ulong) d * sizeof(half);
    for (int jj = 0; jj < jn; ++jj) {
        const float8 vv = (float8) (vload_half(0, (global const half *) (vd + (ulong) jj * nbvj)));
        acc0 = mad(vv, vload8(0, &Pl[jj][0]), acc0);
        acc1 = mad(vv, vload8(1, &Pl[jj][0]), acc1);
        acc2 = mad(vv, vload8(2, &Pl[jj][0]), acc2);
        acc3 = mad(vv, vload8(3, &Pl[jj][0]), acc3);
    }

    global float * out = part + ((ulong) s * NCOL) * D + d;
    NCOL_STORE8(acc0, out, D) out += 8 * D;
    NCOL_STORE8(acc1, out, D) out += 8 * D;
    NCOL_STORE8(acc2, out, D) out += 8 * D;
    NCOL_STORE8(acc3, out, D)
}

kernel void kernel_mul_mat_ncol_split_reduce(
        global const float * part,
        global char * d,
        ulong offd,
        ulong nbd1,
        int n_split,
        int D) {
    const int gid = get_global_id(0);
    if (gid >= NCOL * D) {
        return;
    }
    const int n = gid / D;
    const int dd = gid - n * D;
    float sum = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        sum += part[((ulong) s * NCOL + n) * D + dd];
    }
    *(global float *) (d + offd + (ulong) n * nbd1 + (ulong) dd * sizeof(float)) = sum;
}

#ifdef NCOL_SPLIT_VARIANTS
//------------------------------------------------------------------------------
// Variants of kernel_mul_mat_f16t_f32_ncol_split with two d per lane, without the JC argument (fixed split
// size, full splits only). Built as their own program on Adreno; the host picks one by cache length.
//------------------------------------------------------------------------------
// 64 lanes process 128 adjacent d, all 32 heads; splits of 256 rows
kernel void kernel_mul_mat_f16t_f32_ncol_split_d2(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][36];
    const int lane = get_local_id(0);
    const int d = 2 * (get_group_id(0) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 256 * split;
    float8 even[4] = { 0 }, odd[4] = { 0 };
    for (int strip = 0; strip < 256; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        for (int h = 0; h < 32; ++h) {
            Pl[lane][h] =
                    lane < left ? ((global const float *) (p + offp + (ulong) h * nbp1))[j0 + strip + lane] : 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int u = 0; u < left; ++u) {
            const global half * vp = (global const half *) (v + offv + (ulong) (j0 + strip + u) * nbvj) + d;
            const float2 pair = vload_half2(0, vp);
#pragma unroll
            for (int h8 = 0; h8 < 4; ++h8) {
                const float8 w = vload8(h8, &Pl[u][0]);
                even[h8] = mad((float8) pair.x, w, even[h8]);
                odd[h8] = mad((float8) pair.y, w, odd[h8]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + (ulong) split * 32 * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 4; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}
kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c16(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][20];
    const int lane = get_local_id(0);
    const int head0 = (get_group_id(0) % 2) * 16;
    const int d = 2 * ((get_group_id(0) / 2) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 256 * split;
    float8 even[2] = { 0 }, odd[2] = { 0 };
    for (int strip = 0; strip < 256; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        for (int h = 0; h < 16; ++h) {
            Pl[lane][h] = lane < left ?
                    ((global const float *) (p + offp + (ulong) (head0 + h) * nbp1))[j0 + strip + lane] :
                    0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int u = 0; u < left; ++u) {
            const global half * vp = (global const half *) (v + offv + (ulong) (j0 + strip + u) * nbvj) + d;
            const float2 pair = vload_half2(0, vp);
#pragma unroll
            for (int h8 = 0; h8 < 2; ++h8) {
                const float8 w = vload8(h8, &Pl[u][0]);
                even[h8] = mad((float8) pair.x, w, even[h8]);
                odd[h8] = mad((float8) pair.y, w, odd[h8]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 2; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}
kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c16_u4(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][20];
    const int lane = get_local_id(0);
    const int head0 = (get_group_id(0) % 2) * 16;
    const int d = 2 * ((get_group_id(0) / 2) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 256 * split;
    float8 even[2] = { 0 }, odd[2] = { 0 };
    for (int strip = 0; strip < 256; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        for (int h = 0; h < 16; ++h) {
            Pl[lane][h] = lane < left ?
                    ((global const float *) (p + offp + (ulong) (head0 + h) * nbp1))[j0 + strip + lane] :
                    0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The selector only admits full 256-key splits.
        for (int u = 0; u < left; u += 4) {
            float2 values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const global half * ptr = (global const half *) (v + offv + (ulong) (j0 + strip + u + i) * nbvj) + d;
                values[i] = vload_half2(0, ptr);
            }
#pragma unroll
            for (int h8 = 0; h8 < 2; ++h8) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float8 w = vload8(h8, &Pl[u + i][0]);
                    even[h8] = mad((float8) values[i].x, w, even[h8]);
                    odd[h8] = mad((float8) values[i].y, w, odd[h8]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 2; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}
kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c8_u4(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][12];
    const int lane = get_local_id(0);
    const int head0 = (get_group_id(0) % 4) * 8;
    const int d = 2 * ((get_group_id(0) / 4) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 256 * split;
    float8 even[1] = { 0 }, odd[1] = { 0 };
    for (int strip = 0; strip < 256; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        for (int h = 0; h < 8; ++h) {
            Pl[lane][h] = lane < left ?
                    ((global const float *) (p + offp + (ulong) (head0 + h) * nbp1))[j0 + strip + lane] :
                    0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The selector only admits full 256-key splits.
        for (int u = 0; u < left; u += 4) {
            float2 values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const global half * ptr = (global const half *) (v + offv + (ulong) (j0 + strip + u + i) * nbvj) + d;
                values[i] = vload_half2(0, ptr);
            }
#pragma unroll
            for (int h8 = 0; h8 < 1; ++h8) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float8 w = vload8(h8, &Pl[u + i][0]);
                    even[h8] = mad((float8) values[i].x, w, even[h8]);
                    odd[h8] = mad((float8) values[i].y, w, odd[h8]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 1; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}

kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c8_vp(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][12];
    const int lane = get_local_id(0);
    const int head0 = (get_group_id(0) % 4) * 8;
    const int d = 2 * ((get_group_id(0) / 4) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 256 * split;
    float8 even[1] = { 0 }, odd[1] = { 0 };
    for (int strip = 0; strip < 256; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        const int row = (lane % 16) * 4;
        for (int h = lane / 16; h < 8; h += 4) {
            const global float * weights = (global const float *) (p + offp + (ulong) (head0 + h) * nbp1);
            float4 w = vload4(0, weights + j0 + strip + row);
            Pl[row][h] = w.s0;
            Pl[row + 1][h] = w.s1;
            Pl[row + 2][h] = w.s2;
            Pl[row + 3][h] = w.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The selector only admits full 256-key splits.
        for (int u = 0; u < left; u += 4) {
            float2 values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const global half * ptr = (global const half *) (v + offv + (ulong) (j0 + strip + u + i) * nbvj) + d;
                values[i] = vload_half2(0, ptr);
            }
#pragma unroll
            for (int h8 = 0; h8 < 1; ++h8) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float8 w = vload8(h8, &Pl[u + i][0]);
                    even[h8] = mad((float8) values[i].x, w, even[h8]);
                    odd[h8] = mad((float8) values[i].y, w, odd[h8]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 1; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}

kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c8_vp_j128(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][12];
    const int lane = get_local_id(0);
    const int head0 = (get_group_id(0) % 4) * 8;
    const int d = 2 * ((get_group_id(0) / 4) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 128 * split;
    float8 even[1] = { 0 }, odd[1] = { 0 };
    for (int strip = 0; strip < 128; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        const int row = (lane % 16) * 4;
        for (int h = lane / 16; h < 8; h += 4) {
            const global float * weights = (global const float *) (p + offp + (ulong) (head0 + h) * nbp1);
            float4 w = vload4(0, weights + j0 + strip + row);
            Pl[row][h] = w.s0;
            Pl[row + 1][h] = w.s1;
            Pl[row + 2][h] = w.s2;
            Pl[row + 3][h] = w.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The selector only admits full 128-key splits.
        for (int u = 0; u < left; u += 4) {
            float2 values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const global half * ptr = (global const half *) (v + offv + (ulong) (j0 + strip + u + i) * nbvj) + d;
                values[i] = vload_half2(0, ptr);
            }
#pragma unroll
            for (int h8 = 0; h8 < 1; ++h8) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float8 w = vload8(h8, &Pl[u + i][0]);
                    even[h8] = mad((float8) values[i].x, w, even[h8]);
                    odd[h8] = mad((float8) values[i].y, w, odd[h8]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 1; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}

static inline int pv_j0(int split) {
    return split * 64;
}
static inline int pv_head0(int group) {
    return (group % 4) * 8;
}
static inline int pv_d(int group, int lane) {
    return 2 * ((group / 4) * 64 + lane);
}
static inline int pv_partial(int split, int head, int d) {
    return (split * 32 + head) * 512 + d;
}

kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c8_vp_j64(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D) {
    local float Pl[64][12];
    const int lane = get_local_id(0);
    const int head0 = pv_head0(get_group_id(0));
    const int d = pv_d(get_group_id(0), lane);
    const int split = get_group_id(1);
    const int j0 = pv_j0(split);
    float8 even[1] = { 0 }, odd[1] = { 0 };
    for (int strip = 0; strip < 64; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        const int row = (lane % 16) * 4;
        for (int h = lane / 16; h < 8; h += 4) {
            const global float * weights = (global const float *) (p + offp + (ulong) (head0 + h) * nbp1);
            float4 w = vload4(0, weights + j0 + strip + row);
            Pl[row][h] = w.s0;
            Pl[row + 1][h] = w.s1;
            Pl[row + 2][h] = w.s2;
            Pl[row + 3][h] = w.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The selector only admits full 64-key splits.
        for (int u = 0; u < left; u += 4) {
            float2 values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const global half * ptr = (global const half *) (v + offv + (ulong) (j0 + strip + u + i) * nbvj) + d;
                values[i] = vload_half2(0, ptr);
            }
#pragma unroll
            for (int h8 = 0; h8 < 1; ++h8) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float8 w = vload8(h8, &Pl[u + i][0]);
                    even[h8] = mad((float8) values[i].x, w, even[h8]);
                    odd[h8] = mad((float8) values[i].y, w, odd[h8]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + pv_partial(split, head0, d);
#pragma unroll
    for (int h8 = 0; h8 < 1; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}

//------------------------------------------------------------------------------
// 64 lanes, 8 adjacent d per lane (one half8 load of a cache row per lane and key), 8 heads per group (P of the
// 8 heads for one key read as a float8 from local memory): 64 FMA per load pair. Splits of JC keys; the selector
// only admits full splits (J % JC == 0). Same summation order as the d2 variants.
//------------------------------------------------------------------------------
kernel void kernel_mul_mat_f16t_f32_ncol_split_d8_c8(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D,
        int JC) {
    local float8 Pl[64];
    const int lane = get_local_id(0);
    const int head0 = get_group_id(0) * 8;
    const int d = lane * 8;
    const int split = get_group_id(1);
    const int j0 = JC * split;
    float8 acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f, acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    for (int strip = 0; strip < JC; strip += 64) {
        float w[8];
#pragma unroll
        for (int h = 0; h < 8; ++h) {
            w[h] = ((global const float *) (p + offp + (ulong) (head0 + h) * nbp1))[j0 + strip + lane];
        }
        Pl[lane] = (float8) (w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
        barrier(CLK_LOCAL_MEM_FENCE);
        const global char * base = v + offv + (ulong) (j0 + strip) * nbvj + (ulong) d * sizeof(half);
#pragma unroll 4
        for (int u = 0; u < 64; ++u) {
            const float8 x = convert_float8(vload8(0, (const global half *) (base + (ulong) u * nbvj)));
            const float8 pw = Pl[u];
            acc0 = mad((float8) x.s0, pw, acc0);
            acc1 = mad((float8) x.s1, pw, acc1);
            acc2 = mad((float8) x.s2, pw, acc2);
            acc3 = mad((float8) x.s3, pw, acc3);
            acc4 = mad((float8) x.s4, pw, acc4);
            acc5 = mad((float8) x.s5, pw, acc5);
            acc6 = mad((float8) x.s6, pw, acc6);
            acc7 = mad((float8) x.s7, pw, acc7);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
    global float * o0 = out + 0; NCOL_STORE8(acc0, o0, D)
    global float * o1 = out + 1; NCOL_STORE8(acc1, o1, D)
    global float * o2 = out + 2; NCOL_STORE8(acc2, o2, D)
    global float * o3 = out + 3; NCOL_STORE8(acc3, o3, D)
    global float * o4 = out + 4; NCOL_STORE8(acc4, o4, D)
    global float * o5 = out + 5; NCOL_STORE8(acc5, o5, D)
    global float * o6 = out + 6; NCOL_STORE8(acc6, o6, D)
    global float * o7 = out + 7; NCOL_STORE8(acc7, o7, D)
}

#endif  // NCOL_SPLIT_VARIANTS

#ifdef NCOL_SPLIT_NORM
// The 8-head vector-P split variant that also divides P by the softmax row sums (the softmax then only
// writes exp(x - max) and the sums).
kernel void kernel_mul_mat_f16t_f32_ncol_split_d2_c8_vp_norm(
        global const char * v,
        ulong offv,
        ulong nbvj,
        global const char * p,
        ulong offp,
        ulong nbp1,
        global float * part,
        int J,
        int D,
        global const float * sums) {
    local float Pl[64][12];
    const int lane = get_local_id(0);
    const int head0 = (get_group_id(0) % 4) * 8;
    const int d = 2 * ((get_group_id(0) / 4) * 64 + lane);
    const int split = get_group_id(1);
    const int j0 = 256 * split;
    float8 even[1] = { 0 }, odd[1] = { 0 };
    for (int strip = 0; strip < 256; strip += 64) {
        const int left = clamp(J - j0 - strip, 0, 64);
        const int row = (lane % 16) * 4;
        for (int h = lane / 16; h < 8; h += 4) {
            const global float * weights = (global const float *) (p + offp + (ulong) (head0 + h) * nbp1);
            float4 w = vload4(0, weights + j0 + strip + row) / sums[head0 + h];
            Pl[row][h] = w.s0;
            Pl[row + 1][h] = w.s1;
            Pl[row + 2][h] = w.s2;
            Pl[row + 3][h] = w.s3;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // The selector only admits full 256-key splits.
        for (int u = 0; u < left; u += 4) {
            float2 values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const global half * ptr = (global const half *) (v + offv + (ulong) (j0 + strip + u + i) * nbvj) + d;
                values[i] = vload_half2(0, ptr);
            }
#pragma unroll
            for (int h8 = 0; h8 < 1; ++h8) {
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float8 w = vload8(h8, &Pl[u + i][0]);
                    even[h8] = mad((float8) values[i].x, w, even[h8]);
                    odd[h8] = mad((float8) values[i].y, w, odd[h8]);
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    global float * out = part + ((ulong) split * 32 + head0) * D + d;
#pragma unroll
    for (int h8 = 0; h8 < 1; ++h8) {
        global float * out_odd = out + 1;
        NCOL_STORE8(even[h8], out, D)
        NCOL_STORE8(odd[h8], out_odd, D)
        out += 8 * D;
    }
}
#endif  // NCOL_SPLIT_NORM
