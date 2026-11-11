// POOL_1D (max / avg) over ne0 of each row, F32 -> F32; same loop as the CPU reference.
// one work-item per output element (ow, row)

#pragma OPENCL FP_CONTRACT OFF

kernel void kernel_pool_1d_f32(
        global const char * src,
        ulong offs,
        global char * dst,
        ulong offd,
        int IW,
        int OW,
        int nrows,
        int ne1,
        int ne2,
        ulong nbs1,
        ulong nbs2,
        ulong nbs3,
        ulong nbd1,
        ulong nbd2,
        ulong nbd3,
        int op,  // 0 = max, 1 = avg (GGML_OP_POOL_MAX, GGML_OP_POOL_AVG)
        int k,
        int s,
        int p) {
    const int ow = get_global_id(0);
    const int ir = get_global_id(1);
    if (ow >= OW || ir >= nrows) {
        return;
    }

    const int i1 = ir % ne1;
    const int i2 = (ir / ne1) % ne2;
    const int i3 = ir / (ne1 * ne2);

    global const float * srow = (global const float *) (src + offs + i1 * nbs1 + i2 * nbs2 + i3 * nbs3);
    global float * drow = (global float *) (dst + offd + i1 * nbd1 + i2 * nbd2 + i3 * nbd3);

    float res = op == 0 ? -FLT_MAX : 0.0f;
    int count = 0;
    const int base = ow * s - p;
    for (int ki = 0; ki < k; ++ki) {
        const int j = base + ki;
        if (j < 0 || j >= IW) {
            continue;
        }
        const float v = srow[j];
        // std::max(v, res) returns res unless res < v
        res = op == 0 ? (res < v ? v : res) : res + v;
        ++count;
    }
    if (op == 1) {
        res = count > 0 ? res / count : 0.0f;
    }
    drow[ow] = res;
}
