#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__constant sampler_t smp_zero = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

__attribute__((reqd_work_group_size(16, 16, 1)))
__kernel void adreno_xmem_store_dst_f32_coalesced(
    __read_only image2d_t dst_img,
    __global void * dst_void,
    ulong offset,
    int M,
    int N) {
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);
    const int x = get_group_id(0)*16 + ly;
    const int y = get_group_id(1)*16 + lx;
    const int npack = (M + 3) / 4;
    // Transpose the tile to keep output stores contiguous.
    local half4 tile[16][17];
    tile[ly][lx] = read_imageh(dst_img, smp_zero, (int2)(get_global_id(0), get_global_id(1)));
    barrier(CLK_LOCAL_MEM_FENCE);
    const half4 hv = tile[lx][ly];
    if (x >= N || y >= npack) {
        return;
    }
    __global float * dst = (__global float *)((__global char *)dst_void + offset);
    const int m = y*4;
    if (m + 3 < M) {
        vstore4(convert_float4(hv), 0, dst + x*M + m);
    } else {
        if (m + 0 < M) dst[x*M + m + 0] = (float)hv.s0;
        if (m + 1 < M) dst[x*M + m + 1] = (float)hv.s1;
        if (m + 2 < M) dst[x*M + m + 2] = (float)hv.s2;
    }
}
