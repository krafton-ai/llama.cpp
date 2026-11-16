#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

#define SMALLM_MAX 24
#define SMALLM_NT  2

REQD_SUBGROUP_SIZE_64
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_rms_norm_mul_mm_f16_f32_smallm(
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
        int M,
        int N,
        float eps) {
    const int lid = get_local_id(0);
    const int n0  = get_group_id(0) * SMALLM_NT;

    a += offa;
    b += offb;
    d += offd;

    global const float * b0 = (global const float *) (b + (ulong) min(n0 + 0, N - 1) * nbb1);
    global const float * b1 = (global const float *) (b + (ulong) min(n0 + 1, N - 1) * nbb1);

    float4 sum0 = 0.0f, sum1 = 0.0f;
    for (int k = lid * 4; k < K; k += 256) {
        const float4 v0 = vload4(0, b0 + k);
        const float4 v1 = vload4(0, b1 + k);
        sum0 += v0 * v0;
        sum1 += v1 * v1;
    }
    float s0 = sum0.s0 + sum0.s1 + sum0.s2 + sum0.s3;
    float s1 = sum1.s0 + sum1.s1 + sum1.s2 + sum1.s3;
    s0 = sub_group_reduce_add(s0);
    s1 = sub_group_reduce_add(s1);
    const float scale0 = 1.0f / sqrt(s0 / K + eps);
    const float scale1 = 1.0f / sqrt(s1 / K + eps);

    float acc0[SMALLM_MAX];
    float acc1[SMALLM_MAX];
#pragma unroll
    for (int m = 0; m < SMALLM_MAX; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int k = lid * 4; k < K; k += 256) {
        const float4 x0 = vload4(0, b0 + k) * scale0;
        const float4 x1 = vload4(0, b1 + k) * scale1;
#pragma unroll
        for (int m = 0; m < SMALLM_MAX; ++m) {
            if (m < M) {
                const float4 w = vload_half4(0, (global const half *) (a + (ulong) m * nba1) + k);
                acc0[m] += dot(w, x0);
                acc1[m] += dot(w, x1);
            }
        }
    }

#pragma unroll
    for (int m = 0; m < SMALLM_MAX; ++m) {
        if (m < M) {
            const float s0 = sub_group_reduce_add(acc0[m]);
            const float s1 = sub_group_reduce_add(acc1[m]);
            if (lid == 0) {
                *(global float *) (d + (ulong) (n0 + 0) * nbd1 + (ulong) m * sizeof(float)) = s0;
                if (n0 + 1 < N) {
                    *(global float *) (d + (ulong) (n0 + 1) * nbd1 + (ulong) m * sizeof(float)) = s1;
                }
            }
        }
    }
}
