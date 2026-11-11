#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// cpy
//------------------------------------------------------------------------------

kernel void kernel_cpy_f16_f16(
        global half * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f16_f32(
        global half * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {

    src0 = (global half*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global half * src = (global half *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f16(
        global float * src0,
        ulong offset0,
        global half * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global half*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global half * dst_data = (global half *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

kernel void kernel_cpy_f32_f32(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

#ifndef GGML_CL_NO_CPY_PACK
kernel void kernel_cpy_f32_f32_pack(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst = (global float*)((global char*)dst + offsetd);

    int lsz = get_local_size(0);
    int tpr = min(ne00, lsz);          // threads per row
    int rpw = lsz / tpr;               // rows per workgroup
    int lid = get_local_id(0);
    int row = get_group_id(0)*rpw + lid / tpr;
    int lane = lid - (lid / tpr) * tpr;

    int nrows = ne01*ne02*ne03;
    if (row >= nrows) {
        return;
    }

    int i01 = row % ne01;
    int t   = row / ne01;
    int i02 = t % ne02;
    int i03 = t / ne02;

    // linear index of the first element of this row, unflattened over dst dims
    long n  = (long)row * ne00;
    int i3  = (int)(n / ((long)ne2*ne1*ne0));
    long rm = n - (long)i3*ne2*ne1*ne0;
    int i2  = (int)(rm / ((long)ne1*ne0));
    rm     -= (long)i2*ne1*ne0;
    int i1  = (int)(rm / ne0);
    int i0  = (int)(rm - (long)i1*ne0);

    global float * dst_data = (global float *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = lane; i00 < ne00; i00 += tpr) {
        global const float * src = (global float *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);
        dst_data[i00] = src[0];
    }
}
#endif // GGML_CL_NO_CPY_PACK

kernel void kernel_cpy_i32_i32(
        global int * src0,
        ulong offset0,
        global int * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne03,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        ulong nb03,
        int ne0,
        int ne1,
        int ne2,
        int ne3,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        ulong nb3
) {
    src0 = (global int*)((global char*)src0 + offset0);
    dst = (global int*)((global char*)dst + offsetd);

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0);

    int n = i03*ne02*ne01*ne00 + i02*ne01*ne00 + i01*ne00;

    int i3 = n / (ne2*ne1*ne0);
    int i2 = (n - i3*ne2*ne1*ne0) / (ne1*ne0);
    int i1 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0) / ne0;
    int i0 = (n - i3*ne2*ne1*ne0 - i2*ne1*ne0 - i1*ne0);

    global int * dst_data = (global int *) ((global char *) dst + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

    for (int i00 = get_local_id(0); i00 < ne00; i00 += get_local_size(0)) {
        global const int * src = (global int *)((global char *) src0 + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00);

        dst_data[i00] = src[0];
    }
}

// Contiguous f32 copy, one work item per float4 over the whole tensor. The kernels above map
// one workgroup to each row, which leaves a tensor with few long rows on a single compute unit.
// vload4/vstore4 rather than a float4 cast: these buffers carry an arbitrary 4-byte view offset.
kernel void kernel_cpy_f32_f32_flat(
        global float * src0,
        ulong offset0,
        global float * dst,
        ulong offsetd,
        ulong ne,
        ulong n4
) {
    src0 = (global float*)((global char*)src0 + offset0);
    dst  = (global float*)((global char*)dst  + offsetd);

    const ulong i = get_global_id(0);

    if (i < n4) {
        vstore4(vload4(i, src0), i, dst);
    } else if (i == n4) {
        for (ulong t = n4 * 4; t < ne; ++t) {
            dst[t] = src0[t];
        }
    }
}

// f16 copy of a transposed 2D view (dim 1 of src contiguous, dim 0 strided), e.g. [rank, n_kv] -> [n_kv, rank].
// A 64x64 block goes through local memory so that lanes read and write consecutive halves.
// Work-group (64, 4); the tile row is padded to 66 halves to spread the column reads over the banks.
#define CPY_T2D_TILE 64
kernel void kernel_cpy_f16_f16_t2d(
        global char * src0,
        ulong offset0,
        global char * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        ulong nb00,
        ulong nb02,
        ulong nb03,
        ulong nb1,
        ulong nb2,
        ulong nb3) {
    local half tile[CPY_T2D_TILE][CPY_T2D_TILE + 2];

    const int tx = get_local_id(0);  // 0..63
    const int ty = get_local_id(1);  // 0..3
    const int b1 = get_group_id(0);  // block along dim 1 (contiguous in src)
    const int b0 = get_group_id(1);  // block along dim 0 (contiguous in dst)
    const int i02 = get_group_id(2) % ne02;
    const int i03 = get_group_id(2) / ne02;

    src0 += offset0 + i02 * nb02 + i03 * nb03;
    dst += offsetd + i02 * nb2 + i03 * nb3;

    const int i01 = b1 * CPY_T2D_TILE + tx;
    if (i01 < ne01) {
#pragma unroll
        for (int k = 0; k < CPY_T2D_TILE / 4; ++k) {
            const int j = ty + 4 * k;
            const int i00 = b0 * CPY_T2D_TILE + j;
            if (i00 < ne00) {
                tile[j][tx] = *(global half *) (src0 + i00 * nb00 + i01 * sizeof(half));
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const int i0 = b0 * CPY_T2D_TILE + tx;
    if (i0 < ne00) {
#pragma unroll
        for (int k = 0; k < CPY_T2D_TILE / 4; ++k) {
            const int d = ty + 4 * k;
            const int i1 = b1 * CPY_T2D_TILE + d;
            if (i1 < ne01) {
                *(global half *) (dst + i1 * nb1 + i0 * sizeof(half)) = tile[tx][d];
            }
        }
    }
}
