kernel void kernel_ssm_conv_f32_f32(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2
){
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc  = ne10;

    global float * s = (global float *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float * c = (global float *) (src1 + ir*nb11);
    global float * d = (global float *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc; ++i0) {
        sumf += s[i0] * c[i0];
    }

    d[0] = sumf;
}

kernel void kernel_ssm_conv_f32_f32_4(
    global char * src0,
    ulong         offset0,
    global char * src1,
    ulong         offset1,
    global char * dst,
    ulong         offsetd,
    ulong         nb00,
    ulong         nb01,
    ulong         nb02,
    int           ne10,
    ulong         nb11,
    ulong         nb0,
    ulong         nb1,
    ulong         nb2
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int ir = get_global_id(0);
    int i2 = get_global_id(1);
    int i3 = get_global_id(2);

    int nc = ne10;

    global float4 * s = (global float4 *) (src0 + ir*nb01 + i2*nb00 + i3*nb02);
    global float4 * c = (global float4 *) (src1 + ir*nb11);
    global float  * d = (global float  *) (dst  + ir*nb0  + i2*nb1  + i3*nb2);

    float sumf = 0.0f;

    for (int i0 = 0; i0 < nc/4; ++i0) {
        sumf += dot(s[i0], c[i0]);
    }

    d[0] = sumf;
}

// many tokens: the plain kernels put channels on adjacent work items, so each reads its own row of the
// time-major input and nothing coalesces. Here a 16 channel x 16 token group first copies its input window
// (16 rows of 16 + nc - 1 steps, read along time) to local memory. nc <= SSM_CONV_T_MAXC
#define SSM_CONV_T      16
#define SSM_CONV_T_MAXC 8
kernel void kernel_ssm_conv_f32_f32_t(
        global char * src0,
        ulong offset0,
        global char * src1,
        ulong offset1,
        global char * dst,
        ulong offsetd,
        ulong nb00,
        ulong nb01,
        ulong nb02,
        int ne10,
        ulong nb11,
        ulong nb0,
        ulong nb1,
        ulong nb2,
        int ne01,
        int ne1) {
    local float tile[SSM_CONV_T][SSM_CONV_T + SSM_CONV_T_MAXC];

    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst = dst + offsetd;

    const int nc = ne10;
    const int c0 = get_group_id(0) * SSM_CONV_T;
    const int t0 = get_group_id(1) * SSM_CONV_T;
    const int i3 = get_group_id(2);
    const int tx = get_local_id(0);
    const int ty = get_local_id(1);
    const int w = SSM_CONV_T + nc - 1;

    // the window of token t is input steps t .. t + nc - 1
    for (int j = ty * SSM_CONV_T + tx; j < SSM_CONV_T * w; j += SSM_CONV_T * SSM_CONV_T) {
        const int r = j / w, k = j - r * w;
        const int c = c0 + r, i = t0 + k;
        if (c < ne01 && i < ne1 + nc - 1) {
            tile[r][k] = *(global const float *) (src0 + c * nb01 + i * nb00 + i3 * nb02);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int c = c0 + tx, t = t0 + ty;
    if (c < ne01 && t < ne1) {
        global const float * wc = (global const float *) (src1 + c * nb11);
        float sumf = 0.0f;
        for (int k = 0; k < nc; ++k) {
            sumf += tile[tx][ty + k] * wc[k];
        }
        *(global float *) (dst + c * nb0 + t * nb1 + i3 * nb2) = sumf;
    }
}
