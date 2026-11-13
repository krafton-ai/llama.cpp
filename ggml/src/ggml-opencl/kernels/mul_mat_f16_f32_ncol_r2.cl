// Requires local size 256, M>0, K>0 and K%4==0.
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
#define sub_group_shuffle_xor(v, m) \
    qcom_sub_group_shuffle_xor((v), (m), CLK_SUB_GROUP_SHUFFLE_WIDTH_WAVE_SIZE_QCOM, 0.0f)
#else
#error Required subgroup shuffle extension unavailable
#endif
#define RED8(v, o)                          \
    v.s0 += sub_group_shuffle_xor(v.s0, o); \
    v.s1 += sub_group_shuffle_xor(v.s1, o); \
    v.s2 += sub_group_shuffle_xor(v.s2, o); \
    v.s3 += sub_group_shuffle_xor(v.s3, o); \
    v.s4 += sub_group_shuffle_xor(v.s4, o); \
    v.s5 += sub_group_shuffle_xor(v.s5, o); \
    v.s6 += sub_group_shuffle_xor(v.s6, o); \
    v.s7 += sub_group_shuffle_xor(v.s7, o);
#define STORE8(v, p, s) \
    p[0 * s] = v.s0;    \
    p[1 * s] = v.s1;    \
    p[2 * s] = v.s2;    \
    p[3 * s] = v.s3;    \
    p[4 * s] = v.s4;    \
    p[5 * s] = v.s5;    \
    p[6 * s] = v.s6;    \
    p[7 * s] = v.s7;

#ifdef NCOL_R2
kernel void kernel_mul_mat_f16_f32_ncol_r2(
        global const char * a,
        ulong oa,
        ulong sa,
        global const char * b,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    local half tile[256][32];
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int m0 = get_group_id(0) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f, y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    for (int k0 = 0; k0 < K; k0 += 256) {
        const int n = lid / 8, ks = (lid % 8) * 32;
        const global float * bn = (global const float *) (b + ob + (ulong) n * sb) + k0 + ks;
        for (int i = 0; i < 32; ++i) {
            tile[ks + i][n] = (k0 + ks + i < K) ? (half) bn[i] : (half) 0.0f;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int kend = min(K - k0, 256);
        for (int c = kl * 4; c < kend; c += 32) {
            float4 av0 = convert_float4(vload4(0, a0 + k0 + c));
            float4 av1 = convert_float4(vload4(0, a1 + k0 + c));
#define BOTH(a, b, k)                                       \
    {                                                       \
        float8 q0 = convert_float8(vload8(0, &tile[k][0])); \
        float8 q1 = convert_float8(vload8(1, &tile[k][0])); \
        float8 q2 = convert_float8(vload8(2, &tile[k][0])); \
        float8 q3 = convert_float8(vload8(3, &tile[k][0])); \
        x0 = mad((float8) (a), q0, x0);                     \
        x1 = mad((float8) (a), q1, x1);                     \
        x2 = mad((float8) (a), q2, x2);                     \
        x3 = mad((float8) (a), q3, x3);                     \
        y0 = mad((float8) (b), q0, y0);                     \
        y1 = mad((float8) (b), q1, y1);                     \
        y2 = mad((float8) (b), q2, y2);                     \
        y3 = mad((float8) (b), q3, y3);                     \
    }
            BOTH(av0.s0, av1.s0, c + 0)
            BOTH(av0.s1, av1.s1, c + 1) BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3)
#undef BOTH
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    RED8(x0, 1u)
    RED8(x1, 1u)
    RED8(x2, 1u) RED8(x3, 1u) RED8(y0, 1u) RED8(y1, 1u) RED8(y2, 1u) RED8(y3, 1u) RED8(x0, 2u) RED8(x1, 2u) RED8(x2, 2u)
            RED8(x3, 2u) RED8(y0, 2u) RED8(y1, 2u) RED8(y2, 2u) RED8(y3, 2u) RED8(x0, 4u) RED8(x1, 4u) RED8(x2, 4u)
                    RED8(x3, 4u) RED8(y0, 4u) RED8(y1, 4u) RED8(y2, 4u) RED8(y3, 4u) const int stride =
                            (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride) o += 8 * stride;
        STORE8(x1, o, stride) o += 8 * stride;
        STORE8(x2, o, stride) o += 8 * stride;
        STORE8(x3, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride) o += 8 * stride;
        STORE8(y1, o, stride) o += 8 * stride;
        STORE8(y2, o, stride) o += 8 * stride;
        STORE8(y3, o, stride)
    }
}
#endif  // NCOL_R2

#ifdef NCOL_R2_K576
kernel void kernel_mul_mat_f16_f32_ncol_r2_k576(
        global const char * a,
        ulong oa,
        ulong sa,
        global const char * b,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    local half tile[288][32];
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int m0 = get_group_id(0) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f, y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    for (int k0 = 0; k0 < 576; k0 += 288) {
        const int n = lid / 8, ks = (lid % 8) * 36;
        const global float * bn = (global const float *) (b + ob + (ulong) n * sb) + k0 + ks;
        for (int i = 0; i < 36; ++i) {
            tile[ks + i][n] = (half) bn[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        const int kend = 288;
        for (int c = kl * 4; c < kend; c += 32) {
            float4 av0 = convert_float4(vload4(0, a0 + k0 + c));
            float4 av1 = convert_float4(vload4(0, a1 + k0 + c));
#define BOTH(a, b, k)                                       \
    {                                                       \
        float8 q0 = convert_float8(vload8(0, &tile[k][0])); \
        float8 q1 = convert_float8(vload8(1, &tile[k][0])); \
        float8 q2 = convert_float8(vload8(2, &tile[k][0])); \
        float8 q3 = convert_float8(vload8(3, &tile[k][0])); \
        x0 = mad((float8) (a), q0, x0);                     \
        x1 = mad((float8) (a), q1, x1);                     \
        x2 = mad((float8) (a), q2, x2);                     \
        x3 = mad((float8) (a), q3, x3);                     \
        y0 = mad((float8) (b), q0, y0);                     \
        y1 = mad((float8) (b), q1, y1);                     \
        y2 = mad((float8) (b), q2, y2);                     \
        y3 = mad((float8) (b), q3, y3);                     \
    }
            BOTH(av0.s0, av1.s0, c + 0)
            BOTH(av0.s1, av1.s1, c + 1) BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3)
#undef BOTH
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    RED8(x0, 1u)
    RED8(x1, 1u)
    RED8(x2, 1u) RED8(x3, 1u) RED8(y0, 1u) RED8(y1, 1u) RED8(y2, 1u) RED8(y3, 1u) RED8(x0, 2u) RED8(x1, 2u) RED8(x2, 2u)
            RED8(x3, 2u) RED8(y0, 2u) RED8(y1, 2u) RED8(y2, 2u) RED8(y3, 2u) RED8(x0, 4u) RED8(x1, 4u) RED8(x2, 4u)
                    RED8(x3, 4u) RED8(y0, 4u) RED8(y1, 4u) RED8(y2, 4u) RED8(y3, 4u) const int stride =
                            (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride) o += 8 * stride;
        STORE8(x1, o, stride) o += 8 * stride;
        STORE8(x2, o, stride) o += 8 * stride;
        STORE8(x3, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride) o += 8 * stride;
        STORE8(y1, o, stride) o += 8 * stride;
        STORE8(y2, o, stride) o += 8 * stride;
        STORE8(y3, o, stride)
    }
}
#endif  // NCOL_R2_K576

#ifdef NCOL_R2_PACKED
kernel void kernel_mul_mat_ncol_pack_q(global const char * q, ulong offq, ulong stride, global half * tile) {
    int at = get_global_id(0), k = at / 32, n = at % 32;
    tile[at] = (half) ((global const float *) (q + offq + (ulong) n * stride))[k];
}
kernel void kernel_mul_mat_f16_f32_ncol_r2_k576_pq(
        global const char * a,
        ulong oa,
        ulong sa,
        global const char * b,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    global const half(*tile)[32] = (global const half(*)[32])(b + ob);
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int m0 = get_group_id(0) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f, y0 = 0.0f, y1 = 0.0f, y2 = 0.0f, y3 = 0.0f;
    for (int k0 = 0; k0 < 576; k0 += 288) {
        const int kend = 288;
        for (int c = kl * 4; c < kend; c += 32) {
            float4 av0 = convert_float4(vload4(0, a0 + k0 + c));
            float4 av1 = convert_float4(vload4(0, a1 + k0 + c));
#define BOTH(a, b, k)                                              \
    {                                                              \
        float8 q0 = convert_float8(vload8(0, &tile[k0 + (k)][0])); \
        float8 q1 = convert_float8(vload8(1, &tile[k0 + (k)][0])); \
        float8 q2 = convert_float8(vload8(2, &tile[k0 + (k)][0])); \
        float8 q3 = convert_float8(vload8(3, &tile[k0 + (k)][0])); \
        x0 = mad((float8) (a), q0, x0);                            \
        x1 = mad((float8) (a), q1, x1);                            \
        x2 = mad((float8) (a), q2, x2);                            \
        x3 = mad((float8) (a), q3, x3);                            \
        y0 = mad((float8) (b), q0, y0);                            \
        y1 = mad((float8) (b), q1, y1);                            \
        y2 = mad((float8) (b), q2, y2);                            \
        y3 = mad((float8) (b), q3, y3);                            \
    }
            BOTH(av0.s0, av1.s0, c + 0)
            BOTH(av0.s1, av1.s1, c + 1) BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3)
#undef BOTH
        }
    }
    RED8(x0, 1u)
    RED8(x1, 1u)
    RED8(x2, 1u) RED8(x3, 1u) RED8(y0, 1u) RED8(y1, 1u) RED8(y2, 1u) RED8(y3, 1u) RED8(x0, 2u) RED8(x1, 2u) RED8(x2, 2u)
            RED8(x3, 2u) RED8(y0, 2u) RED8(y1, 2u) RED8(y2, 2u) RED8(y3, 2u) RED8(x0, 4u) RED8(x1, 4u) RED8(x2, 4u)
                    RED8(x3, 4u) RED8(y0, 4u) RED8(y1, 4u) RED8(y2, 4u) RED8(y3, 4u) const int stride =
                            (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride) o += 8 * stride;
        STORE8(x1, o, stride) o += 8 * stride;
        STORE8(x2, o, stride) o += 8 * stride;
        STORE8(x3, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride) o += 8 * stride;
        STORE8(y1, o, stride) o += 8 * stride;
        STORE8(y2, o, stride) o += 8 * stride;
        STORE8(y3, o, stride)
    }
}

kernel void kernel_mul_mat_f16_f32_ncol_r2_k576_pq_c16(
        global const char * a,
        ulong oa,
        ulong sa,
        global const char * b,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    global const half(*tile)[32] = (global const half(*)[32])(b + ob);
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int head0 = (get_group_id(0) % 2) * 16;
    const int m0 = (get_group_id(0) / 2) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, x1 = 0.0f, y0 = 0.0f, y1 = 0.0f;
    for (int k0 = 0; k0 < 576; k0 += 288) {
        const int kend = 288;
        for (int c = kl * 4; c < kend; c += 32) {
            float4 av0 = convert_float4(vload4(0, a0 + k0 + c));
            float4 av1 = convert_float4(vload4(0, a1 + k0 + c));
#define BOTH(a, b, k)                                                  \
    {                                                                  \
        float8 q0 = convert_float8(vload8(0, &tile[k0 + (k)][head0])); \
        float8 q1 = convert_float8(vload8(1, &tile[k0 + (k)][head0])); \
        x0 = mad((float8) (a), q0, x0);                                \
        x1 = mad((float8) (a), q1, x1);                                \
        y0 = mad((float8) (b), q0, y0);                                \
        y1 = mad((float8) (b), q1, y1);                                \
    }
            BOTH(av0.s0, av1.s0, c + 0)
            BOTH(av0.s1, av1.s1, c + 1) BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3)
#undef BOTH
        }
    }
    RED8(x0, 1u)
    RED8(x1, 1u)
    RED8(y0, 1u) RED8(y1, 1u) RED8(x0, 2u) RED8(x1, 2u) RED8(y0, 2u) RED8(y1, 2u) RED8(x0, 4u) RED8(x1, 4u) RED8(y0, 4u)
            RED8(y1, 4u) const int stride = (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride) o += 8 * stride;
        STORE8(x1, o, stride) o += 8 * stride;
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride) o += 8 * stride;
        STORE8(y1, o, stride) o += 8 * stride;
    }
}

kernel void kernel_mul_mat_f16_f32_ncol_r2_k576_pq_c8(
        global const char * a,
        ulong oa,
        ulong sa,
        global const char * b,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    global const half(*tile)[32] = (global const half(*)[32])(b + ob);
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int head0 = (get_group_id(0) % 4) * 8;
    const int m0 = (get_group_id(0) / 4) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, y0 = 0.0f;
    for (int k0 = 0; k0 < 576; k0 += 288) {
        const int kend = 288;
        for (int c = kl * 4; c < kend; c += 32) {
            float4 av0 = convert_float4(vload4(0, a0 + k0 + c));
            float4 av1 = convert_float4(vload4(0, a1 + k0 + c));
#define BOTH(a, b, k)                                                  \
    {                                                                  \
        float8 q0 = convert_float8(vload8(0, &tile[k0 + (k)][head0])); \
        x0 = mad((float8) (a), q0, x0);                                \
        y0 = mad((float8) (b), q0, y0);                                \
    }
            BOTH(av0.s0, av1.s0, c + 0)
            BOTH(av0.s1, av1.s1, c + 1) BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3)
#undef BOTH
        }
    }
    RED8(x0, 1u)
    RED8(y0, 1u)
    RED8(x0, 2u)
    RED8(y0, 2u)
    RED8(x0, 4u)
    RED8(y0, 4u)
    const int stride = (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride)
    }
}

kernel void kernel_mul_mat_f16_f32_ncol_r2_k576_pq_c8_img(
        global const char * a,
        ulong oa,
        ulong sa,
        read_only image1d_buffer_t qimage,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int head0 = (get_group_id(0) % 4) * 8;
    const int m0 = (get_group_id(0) / 4) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, y0 = 0.0f;
    for (int k0 = 0; k0 < 576; k0 += 288) {
        const int kend = 288;
        for (int c = kl * 4; c < kend; c += 32) {
            float4 av0 = convert_float4(vload4(0, a0 + k0 + c));
            float4 av1 = convert_float4(vload4(0, a1 + k0 + c));
#define BOTH(a, b, k)                                                                      \
    {                                                                                      \
        int pixel = (k0 + (k)) * 8 + head0 / 4;                                            \
        float8 q0 = (float8) (read_imagef(qimage, pixel), read_imagef(qimage, pixel + 1)); \
        x0 = mad((float8) (a), q0, x0);                                                    \
        y0 = mad((float8) (b), q0, y0);                                                    \
    }
            BOTH(av0.s0, av1.s0, c + 0)
            BOTH(av0.s1, av1.s1, c + 1) BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3)
#undef BOTH
        }
    }
    RED8(x0, 1u)
    RED8(y0, 1u)
    RED8(x0, 2u)
    RED8(y0, 2u)
    RED8(x0, 4u)
    RED8(y0, 4u)
    const int stride = (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride)
    }
}

kernel void kernel_mul_mat_f16_f32_ncol_r2_k576_pq_c8_img2(
        global const char * a,
        ulong oa,
        ulong sa,
        read_only image1d_buffer_t qimage,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int head0 = (get_group_id(0) % 4) * 8;
    const int m0 = (get_group_id(0) / 4) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, y0 = 0.0f;
    for (int c = kl * 4; c < 576; c += 64) {
        float4 av0 = convert_float4(vload4(0, a0 + c));
        float4 av1 = convert_float4(vload4(0, a1 + c));
        float4 bv0 = convert_float4(vload4(0, a0 + c + 32));
        float4 bv1 = convert_float4(vload4(0, a1 + c + 32));
#define BOTH(a, b, k)                                                                      \
    {                                                                                      \
        int pixel = (k) * 8 + head0 / 4;                                                   \
        float8 q0 = (float8) (read_imagef(qimage, pixel), read_imagef(qimage, pixel + 1)); \
        x0 = mad((float8) (a), q0, x0);                                                    \
        y0 = mad((float8) (b), q0, y0);                                                    \
    }
        BOTH(av0.s0, av1.s0, c + 0)
        BOTH(av0.s1, av1.s1, c + 1)
        BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3) BOTH(bv0.s0, bv1.s0, c + 32)
                BOTH(bv0.s1, bv1.s1, c + 33) BOTH(bv0.s2, bv1.s2, c + 34) BOTH(bv0.s3, bv1.s3, c + 35)
#undef BOTH
    }
    RED8(x0, 1u)
    RED8(y0, 1u)
    RED8(x0, 2u)
    RED8(y0, 2u)
    RED8(x0, 4u)
    RED8(y0, 4u)
    const int stride = (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride)
    }
}

// img2 with q viewed as 128-bit texels (CL_RGBA / CL_UNSIGNED_INT32): one fetch gives the 8 heads of one k
kernel void kernel_mul_mat_f16_f32_ncol_r2_k576_pq_c8_img4(
        global const char * a,
        ulong oa,
        ulong sa,
        read_only image1d_buffer_t qimage,
        ulong ob,
        ulong sb,
        global char * d,
        ulong od,
        ulong sd,
        int K,
        int M) {
    const int lid = get_local_id(0), rl = lid / 8, kl = lid % 8;
    const int head0 = (get_group_id(0) % 4) * 8;
    const int m0 = (get_group_id(0) / 4) * 64 + rl, m1 = m0 + 32;
    const global half * a0 = (global const half *) (a + oa + (ulong) min(m0, M - 1) * sa);
    const global half * a1 = (global const half *) (a + oa + (ulong) min(m1, M - 1) * sa);
    float8 x0 = 0.0f, y0 = 0.0f;
    for (int c = kl * 4; c < 576; c += 64) {
        float4 av0 = convert_float4(vload4(0, a0 + c));
        float4 av1 = convert_float4(vload4(0, a1 + c));
        float4 bv0 = convert_float4(vload4(0, a0 + c + 32));
        float4 bv1 = convert_float4(vload4(0, a1 + c + 32));
#define BOTH(a, b, k)                                                                    \
    {                                                                                    \
        float8 q0 = convert_float8(as_half8(read_imageui(qimage, (k) * 4 + head0 / 8))); \
        x0 = mad((float8) (a), q0, x0);                                                  \
        y0 = mad((float8) (b), q0, y0);                                                  \
    }
        BOTH(av0.s0, av1.s0, c + 0)
        BOTH(av0.s1, av1.s1, c + 1)
        BOTH(av0.s2, av1.s2, c + 2) BOTH(av0.s3, av1.s3, c + 3) BOTH(bv0.s0, bv1.s0, c + 32)
                BOTH(bv0.s1, bv1.s1, c + 33) BOTH(bv0.s2, bv1.s2, c + 34) BOTH(bv0.s3, bv1.s3, c + 35)
#undef BOTH
    }
    RED8(x0, 1u)
    RED8(y0, 1u)
    RED8(x0, 2u)
    RED8(y0, 2u)
    RED8(x0, 4u)
    RED8(y0, 4u)
    const int stride = (int) (sd / sizeof(float));
    if (kl == 0 && m0 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m0 * sizeof(float));
        STORE8(x0, o, stride)
    }
    if (kl == 0 && m1 < M) {
        global float * o = (global float *) (d + od + (ulong) head0 * sd + (ulong) m1 * sizeof(float));
        STORE8(y0, o, stride)
    }
}
#endif  // NCOL_R2_PACKED
