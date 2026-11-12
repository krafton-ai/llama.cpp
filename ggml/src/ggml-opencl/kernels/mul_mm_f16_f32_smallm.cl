#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#else
#define REQD_SUBGROUP_SIZE_64
#endif

// D[n][m] = sum_k A[m][k] * B[n][k] for an f16 A with few rows (M <= SMALLM_MAX) and many columns n, accumulated
// in f32. The tiled GEMM spends a whole 64-row tile on such an A and runs only ne11/64 work-groups over the full K;
// here one 64-lane work-group (one subgroup) takes SMALLM_NT columns and all M rows, each lane a 4-wide slice of K
// every 256, and the lanes are summed with subgroup reductions.
#define SMALLM_MAX 24
#define SMALLM_NT  2

REQD_SUBGROUP_SIZE_64
__attribute__((reqd_work_group_size(64, 1, 1)))
kernel void kernel_mul_mm_f16_f32_smallm(
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
        int N) {
    const int lid = get_local_id(0);
    const int n0  = get_group_id(0) * SMALLM_NT;

    a += offa;
    b += offb;
    d += offd;

    global const float * b0 = (global const float *) (b + (ulong) min(n0 + 0, N - 1) * nbb1);
    global const float * b1 = (global const float *) (b + (ulong) min(n0 + 1, N - 1) * nbb1);

    float acc0[SMALLM_MAX];
    float acc1[SMALLM_MAX];
#pragma unroll
    for (int m = 0; m < SMALLM_MAX; ++m) {
        acc0[m] = 0.0f;
        acc1[m] = 0.0f;
    }

    for (int k = lid * 4; k < K; k += 256) {
        const float4 x0 = vload4(0, b0 + k);
        const float4 x1 = vload4(0, b1 + k);
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
