#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#define LM_FIRST_256B   0
#define LM_SECOND_256B  64
#define LM_THIRD_256B   128
#define LM_FOURTH_256B  192


inline float16 mm_load_a(
    image1d_buffer_t matrix_A,
    uint subMatrixAStartInElements,
    int nb01,
    int line_stride_matrix_A_in_bytes
) {
    __private float8 regA;
    size_t sub_block_id_m = get_local_id(0);

#ifdef KQV
    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * nb01/4);
#else // KQ
    uint a_texCoord = subMatrixAStartInElements/2 + (sub_block_id_m * line_stride_matrix_A_in_bytes/4);
#endif

    regA.s0123  = read_imagef(matrix_A, a_texCoord/4);
    regA.s4567  = read_imagef(matrix_A, (a_texCoord+4)/4);

    return convert_float16(as_half16(regA));
}

inline float4 alu_32(
    float16 regA,
    __local float4* matrix_B_vec
) {

    __private float4 rC = 0;
    int i = get_sub_group_id() * 64;

    rC += regA.s0  * matrix_B_vec[i];
    rC += regA.s1  * matrix_B_vec[i + 16];
    rC += regA.s4  * matrix_B_vec[i + 1];
    rC += regA.s5  * matrix_B_vec[i + 17];
    rC += regA.s8  * matrix_B_vec[i + 2];
    rC += regA.s9  * matrix_B_vec[i + 18];
    rC += regA.sc  * matrix_B_vec[i + 3];
    rC += regA.sd  * matrix_B_vec[i + 19];

    i += 32;

    rC += regA.s2  * matrix_B_vec[i];
     rC += regA.s3  * matrix_B_vec[i + 16];
    rC += regA.s6  * matrix_B_vec[i + 1];
    rC += regA.s7  * matrix_B_vec[i + 17];
    rC += regA.sa  * matrix_B_vec[i + 2];
    rC += regA.sb  * matrix_B_vec[i + 18];
    rC += regA.se  * matrix_B_vec[i + 3];
    rC += regA.sf  * matrix_B_vec[i + 19];

    return rC;
}

inline float16 alu_16(
    float16 regA,
    __local float* matrix_B_local
) {
    float16 out;
    __local float4* matrix_B_vec = (__local float4*)matrix_B_local;

    out.s0123 = alu_32(regA, matrix_B_vec);
    out.s4567 = alu_32(regA, matrix_B_vec + 4);
    out.s89ab = alu_32(regA, matrix_B_vec + 8);
    out.scdef = alu_32(regA, matrix_B_vec + 12);

    return out;
}

// Reuse each shared FP32 query vector for two key tiles. Each accumulator
// retains the public kernel's multiplication and addition order.
inline void alu_32_pair(float16 a, float16 b, __local float4 *v, __private float4 *x, __private float4 *y) {
    float4 r0 = (float4)(0.0f), r1 = (float4)(0.0f);
    int i = get_sub_group_id() * 64;
    float4 z;
    z = v[i + 0];
    r0 += a.s0 * z;
    r1 += b.s0 * z;
    z = v[i + 16];
    r0 += a.s1 * z;
    r1 += b.s1 * z;
    z = v[i + 1];
    r0 += a.s4 * z;
    r1 += b.s4 * z;
    z = v[i + 17];
    r0 += a.s5 * z;
    r1 += b.s5 * z;
    z = v[i + 2];
    r0 += a.s8 * z;
    r1 += b.s8 * z;
    z = v[i + 18];
    r0 += a.s9 * z;
    r1 += b.s9 * z;
    z = v[i + 3];
    r0 += a.sc * z;
    r1 += b.sc * z;
    z = v[i + 19];
    r0 += a.sd * z;
    r1 += b.sd * z;
    z = v[i + 32];
    r0 += a.s2 * z;
    r1 += b.s2 * z;
    z = v[i + 48];
    r0 += a.s3 * z;
    r1 += b.s3 * z;
    z = v[i + 33];
    r0 += a.s6 * z;
    r1 += b.s6 * z;
    z = v[i + 49];
    r0 += a.s7 * z;
    r1 += b.s7 * z;
    z = v[i + 34];
    r0 += a.sa * z;
    r1 += b.sa * z;
    z = v[i + 50];
    r0 += a.sb * z;
    r1 += b.sb * z;
    z = v[i + 35];
    r0 += a.se * z;
    r1 += b.se * z;
    z = v[i + 51];
    r0 += a.sf * z;
    r1 += b.sf * z;
    *x = r0;
    *y = r1;
}
inline void alu_16_pair(float16 a, float16 b, __local float *m, __private float16 *x, __private float16 *y) {
    __local float4 *v = (__local float4 *)m;
    float4 u, w;
    alu_32_pair(a, b, v + 0, &u, &w);
    (*x).s0123 = u;
    (*y).s0123 = w;
    alu_32_pair(a, b, v + 4, &u, &w);
    (*x).s4567 = u;
    (*y).s4567 = w;
    alu_32_pair(a, b, v + 8, &u, &w);
    (*x).s89ab = u;
    (*y).s89ab = w;
    alu_32_pair(a, b, v + 12, &u, &w);
    (*x).scdef = u;
    (*y).scdef = w;
}
inline void mm_mad_pair(__local float *m, float16 a, float16 b, float8 q, uint bo, __private float16 *c0,
                        __private float16 *c1, __private float16 *d0, __private float16 *d1) {
    int o = bo + get_sub_group_id() * 256;
    float16 x, y;
    m[o] = q.s0;
    m[o + 64] = q.s1;
    m[o + 128] = q.s2;
    m[o + 192] = q.s3;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    alu_16_pair(a, b, m, &x, &y);
    *c0 += x;
    *d0 += y;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    m[o] = q.s4;
    m[o + 64] = q.s5;
    m[o + 128] = q.s6;
    m[o + 192] = q.s7;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
    alu_16_pair(a, b, m, &x, &y);
    *c1 += x;
    *d1 += y;
    sub_group_barrier(CLK_LOCAL_MEM_FENCE);
}
inline void mm_store_c_N(__write_only image1d_buffer_t matrix_C, float16 regC0, float16 regC1,
                         uint subMatrixCStartInElements, int line_stride_matrix_C_in_bytes, int mask) {
    size_t sub_block_id_m = get_local_id(0);

    uint strideInWords     = line_stride_matrix_C_in_bytes/4;
    uint c_coordInWords_0  = (subMatrixCStartInElements + sub_block_id_m);

    uint c_coordInWords_1  = c_coordInWords_0 + 1  * strideInWords;
    uint c_coordInWords_2  = c_coordInWords_0 + 2  * strideInWords;
    uint c_coordInWords_3  = c_coordInWords_0 + 3  * strideInWords;
    uint c_coordInWords_4  = c_coordInWords_0 + 4  * strideInWords;
    uint c_coordInWords_5  = c_coordInWords_0 + 5  * strideInWords;
    uint c_coordInWords_6  = c_coordInWords_0 + 6  * strideInWords;
    uint c_coordInWords_7  = c_coordInWords_0 + 7  * strideInWords;
    uint c_coordInWords_8  = c_coordInWords_0 + 8  * strideInWords;
    uint c_coordInWords_9  = c_coordInWords_0 + 9  * strideInWords;
    uint c_coordInWords_10 = c_coordInWords_0 + 10 * strideInWords;
    uint c_coordInWords_11 = c_coordInWords_0 + 11 * strideInWords;
    uint c_coordInWords_12 = c_coordInWords_0 + 12 * strideInWords;
    uint c_coordInWords_13 = c_coordInWords_0 + 13 * strideInWords;
    uint c_coordInWords_14 = c_coordInWords_0 + 14 * strideInWords;
    uint c_coordInWords_15 = c_coordInWords_0 + 15 * strideInWords;
    uint c_coordInWords_16 = c_coordInWords_0 + 16 * strideInWords;
    uint c_coordInWords_17 = c_coordInWords_0 + 17 * strideInWords;
    uint c_coordInWords_18 = c_coordInWords_0 + 18 * strideInWords;
    uint c_coordInWords_19 = c_coordInWords_0 + 19 * strideInWords;
    uint c_coordInWords_20 = c_coordInWords_0 + 20 * strideInWords;
    uint c_coordInWords_21 = c_coordInWords_0 + 21 * strideInWords;
    uint c_coordInWords_22 = c_coordInWords_0 + 22 * strideInWords;
    uint c_coordInWords_23 = c_coordInWords_0 + 23 * strideInWords;
    uint c_coordInWords_24 = c_coordInWords_0 + 24 * strideInWords;
    uint c_coordInWords_25 = c_coordInWords_0 + 25 * strideInWords;
    uint c_coordInWords_26 = c_coordInWords_0 + 26 * strideInWords;
    uint c_coordInWords_27 = c_coordInWords_0 + 27 * strideInWords;
    uint c_coordInWords_28 = c_coordInWords_0 + 28 * strideInWords;
    uint c_coordInWords_29 = c_coordInWords_0 + 29 * strideInWords;
    uint c_coordInWords_30 = c_coordInWords_0 + 30 * strideInWords;
    uint c_coordInWords_31 = c_coordInWords_0 + 31 * strideInWords;

    if (mask > 0)  { write_imagef(matrix_C, c_coordInWords_0, regC0.s0);  }
    if (mask > 1)  { write_imagef(matrix_C, c_coordInWords_1, regC0.s1);  }
    if (mask > 2)  { write_imagef(matrix_C, c_coordInWords_2, regC0.s2);  }
    if (mask > 3)  { write_imagef(matrix_C, c_coordInWords_3, regC0.s3);  }
    if (mask > 4)  { write_imagef(matrix_C, c_coordInWords_4, regC0.s4);  }
    if (mask > 5)  { write_imagef(matrix_C, c_coordInWords_5, regC0.s5);  }
    if (mask > 6)  { write_imagef(matrix_C, c_coordInWords_6, regC0.s6);  }
    if (mask > 7)  { write_imagef(matrix_C, c_coordInWords_7, regC0.s7);  }
    if (mask > 8)  { write_imagef(matrix_C, c_coordInWords_8, regC0.s8);  }
    if (mask > 9)  { write_imagef(matrix_C, c_coordInWords_9, regC0.s9);  }
    if (mask > 10) { write_imagef(matrix_C, c_coordInWords_10, regC0.sa); }
    if (mask > 11) { write_imagef(matrix_C, c_coordInWords_11, regC0.sb); }
    if (mask > 12) { write_imagef(matrix_C, c_coordInWords_12, regC0.sc); }
    if (mask > 13) { write_imagef(matrix_C, c_coordInWords_13, regC0.sd); }
    if (mask > 14) { write_imagef(matrix_C, c_coordInWords_14, regC0.se); }
    if (mask > 15) { write_imagef(matrix_C, c_coordInWords_15, regC0.sf); }
    if (mask > 16) { write_imagef(matrix_C, c_coordInWords_16, regC1.s0); }
    if (mask > 17) { write_imagef(matrix_C, c_coordInWords_17, regC1.s1); }
    if (mask > 18) { write_imagef(matrix_C, c_coordInWords_18, regC1.s2); }
    if (mask > 19) { write_imagef(matrix_C, c_coordInWords_19, regC1.s3); }
    if (mask > 20) { write_imagef(matrix_C, c_coordInWords_20, regC1.s4); }
    if (mask > 21) { write_imagef(matrix_C, c_coordInWords_21, regC1.s5); }
    if (mask > 22) { write_imagef(matrix_C, c_coordInWords_22, regC1.s6); }
    if (mask > 23) { write_imagef(matrix_C, c_coordInWords_23, regC1.s7); }
    if (mask > 24) { write_imagef(matrix_C, c_coordInWords_24, regC1.s8); }
    if (mask > 25) { write_imagef(matrix_C, c_coordInWords_25, regC1.s9); }
    if (mask > 26) { write_imagef(matrix_C, c_coordInWords_26, regC1.sa); }
    if (mask > 27) { write_imagef(matrix_C, c_coordInWords_27, regC1.sb); }
    if (mask > 28) { write_imagef(matrix_C, c_coordInWords_28, regC1.sc); }
    if (mask > 29) { write_imagef(matrix_C, c_coordInWords_29, regC1.sd); }
    if (mask > 30) { write_imagef(matrix_C, c_coordInWords_30, regC1.se); }
    if (mask > 31) { write_imagef(matrix_C, c_coordInWords_31, regC1.sf); }
}

#define TILESIZE_K 16
#define TILESIZE_M 64
#define TILESIZE_N 32
#ifdef KQV
__kernel void mul_mm_f16_f32_kqv(
#else
__kernel void mla_qk_prefill(
#endif
        __read_only  image1d_buffer_t matrix_A,
        int offset0,
        __global float* matrix_B,
        int offset1,
        __write_only image1d_buffer_t matrix_C,
        int offsetd,
        int M, int K, int N,
        int D_A,
        int D_B,
        int nb01
) {
    // The host qualifies the subgroup layout using this same compiled entry.
    // No attention output is computed for the sentinel offset.
    if (offsetd == -1) {
        int i = get_local_id(0) + 64 * get_local_id(2);
        write_imagef(matrix_C, 4 * i, convert_float(get_sub_group_size()));
        write_imagef(matrix_C, 4 * i + 1, convert_float(get_num_sub_groups()));
        write_imagef(matrix_C, 4 * i + 2, convert_float(get_sub_group_id()));
        write_imagef(matrix_C, 4 * i + 3, convert_float(get_sub_group_local_id()));
        return;
    }

    uint block_id_m = get_global_id(1);
    uint block_id_n = get_global_id(2) % ((N+TILESIZE_N-1)/TILESIZE_N);
    uint block_id_d = get_global_id(2) / ((N+TILESIZE_N-1)/TILESIZE_N);

    __private float16  regA;
    __private float8   regB;
    __private float16 regA1;
    __private float16 regD0 = (float16)(0.0f);
    __private float16 regD1 = (float16)(0.0f);
    __private float16 regC0 = (float16)(0.0f);
    __private float16 regC1 = (float16)(0.0f);

    const uint col   = block_id_m * (2*TILESIZE_M);
    const uint row   = block_id_n * TILESIZE_N;
    const uint depth_A = block_id_d / (D_B/D_A);
    const uint depth_B = block_id_d;

#ifdef KQV
    int line_stride_matrix_A_in_bytes = nb01 * M;
    int line_stride_matrix_B_in_bytes = K * N * 4;
#else
    int line_stride_matrix_A_in_bytes = K * D_A * 2;
    int line_stride_matrix_B_in_bytes = K * D_B * 4;
#endif

    int line_stride_matrix_C_in_bytes = M * 4;

    const uint strideAinElements = line_stride_matrix_A_in_bytes / 2;
    const uint strideBinElements = line_stride_matrix_B_in_bytes / 4;

    size_t sub_block_id_m = get_local_id(0);

    uint b_localOffsetInWords = (sub_block_id_m/16)*16
                           + ((((sub_block_id_m)>>0)&1)<<2)
                           + ((((sub_block_id_m)>>1)&1)<<3)
                           + ((((sub_block_id_m)>>2)&1)<<0)
                           + ((((sub_block_id_m)>>3)&1)<<1);

    uint2 b_globalOffsetInWords_xy = {((sub_block_id_m%4)*4), (sub_block_id_m>>2)};
    uint b_globalOffsetInWords00, b_globalOffsetInWords16;
#ifdef KQV
    b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*K;
    b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * K);
    uint subMatrixAStartInElements = depth_A * strideAinElements + col * nb01 / 2;
    uint subMatrixBStartInElements = depth_B * strideBinElements + row * K;
#else
    b_globalOffsetInWords00 = b_globalOffsetInWords_xy.x + b_globalOffsetInWords_xy.y*line_stride_matrix_B_in_bytes/4;
    b_globalOffsetInWords16 = b_globalOffsetInWords00 + (16 * line_stride_matrix_B_in_bytes/4);
    uint subMatrixAStartInElements = col * strideAinElements + depth_A * K;
    uint subMatrixBStartInElements = row * strideBinElements + depth_B * K;
#endif

    __local float matrix_B_local[1024];

    for (uint step=0; step < K; step+=TILESIZE_K) {
        size_t sub_block_id_m = get_local_id(0);
        regA = mm_load_a(matrix_A, subMatrixAStartInElements, nb01, line_stride_matrix_A_in_bytes);
        regA1 = mm_load_a(matrix_A, subMatrixAStartInElements + 64*strideAinElements, nb01, line_stride_matrix_A_in_bytes);

        uint b_coordInWords00 = subMatrixBStartInElements + b_globalOffsetInWords00;
        uint b_coordInWords16 = subMatrixBStartInElements + b_globalOffsetInWords16;

        regB.s0123 = vload4(b_coordInWords00/4, matrix_B);
        regB.s4567 = vload4(b_coordInWords16/4, matrix_B);

        mm_mad_pair(matrix_B_local, regA, regA1, regB, b_localOffsetInWords, &regC0, &regC1, &regD0, &regD1);

        subMatrixAStartInElements += TILESIZE_K;
        subMatrixBStartInElements += TILESIZE_K;
    }

    uint subMatrixCStartInElements = depth_B * N * M + row * M + col;
    mm_store_c_N(matrix_C, regC0, regC1, subMatrixCStartInElements, line_stride_matrix_C_in_bytes, (N-block_id_n*32));
    mm_store_c_N(matrix_C, regD0, regD1, subMatrixCStartInElements+64, line_stride_matrix_C_in_bytes, (N-block_id_n*32));
}
