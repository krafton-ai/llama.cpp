#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void
adreno_xmem_pack_src_f32_k(__global const void * src_void, ulong offset, __write_only image2d_t src_img, int K, int N) {
    const int x = get_global_id(1);
    const int y = get_global_id(0);
    const int kpack = K / 4;

    if (x >= N || y >= kpack) {
        return;
    }

    __global const float * src = (__global const float *) ((__global const char *) src_void + offset);
    const int base = x * K + y * 4;
    const half4 v = (half4) ((half) src[base + 0], (half) src[base + 1], (half) src[base + 2], (half) src[base + 3]);
    write_imageh(src_img, (int2) (x, y), v);
}
