kernel void kernel_ssm_conv_silu_prefill(
        global const char * state, ulong offset_s,
        global const char * x, ulong offset_x,
        global const char * w, ulong offset_w,
        global char * y, ulong offset_y,
        int C, int T, int W) {
    const int c = get_global_id(0), t = get_global_id(1);
    if (c >= C || t >= T) {
        return;
    }
    global const float * sp = (global const float *) (state + offset_s);
    global const float * xp = (global const float *) (x + offset_x);
    global const float * wp = (global const float *) (w + offset_w);
    global float * yp = (global float *) (y + offset_y);
    float acc = 0.0f;
    for (int k = 0; k < W; ++k) {
        const int i = t + k - (W - 1);
        const float v = i < 0 ? sp[c * (W - 1) + t + k] : xp[i * C + c];
        acc += v * wp[c * W + k];
    }
    yp[t * C + c] = acc / (1.0f + exp(-acc));
}

kernel void kernel_ssm_conv_copy_input(
        global const char * x, ulong offset_x,
        global char * tmp, ulong offset_tmp,
        int C, int T, int pitch) {
    const int c = get_global_id(0), t = get_global_id(1);
    if (c < C && t < T) {
        global const float * xp = (global const float *) (x + offset_x);
        global float * tp = (global float *) (tmp + offset_tmp);
        tp[t * C + c] = xp[t * pitch + c];
    }
}

kernel void kernel_ssm_conv_copy_state(
        global const char * x, ulong offset_x,
        global char * dst, ulong offset_dst,
        int C, int T, int W) {
    const int c = get_global_id(0);
    if (c < C) {
        global const float * xp = (global const float *) (x + offset_x);
        global float * dp = (global float *) (dst + offset_dst);
        for (int k = 0; k < W - 1; ++k) {
            dp[c * (W - 1) + k] = xp[(T - (W - 1) + k) * C + c];
        }
    }
}
