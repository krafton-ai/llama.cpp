#extension GL_EXT_control_flow_attributes : require
#if USE_SUBGROUP_ADD
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif


// Shared by the fusion shaders (hc_mix, hc_select, hc_pre_norm, ssm_conv_state). Field meaning per shader is listed in each;
// strides and offsets are in elements, a/b/c/e are the sources in order, d the destination.
layout(push_constant) uniform parameter
{
    uint ne0; uint ne1; uint ne2; uint ne3;
    float f0; float f1;
    uint a_nb0; uint a_nb1; uint a_nb2;
    uint b_nb0; uint b_nb1; uint b_nb2;
    uint c_nb1;
    uint e_nb1; uint e_nb2;
    uint d_nb1; uint d_nb2;
    uint a_off; uint b_off; uint c_off; uint e_off; uint d_off;
    uint o_off; uint o_nb2;    // a second output
};

layout(constant_id = 0) const uint BLOCK_SIZE = 256;
layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;

shared float red[BLOCK_SIZE];

// workgroup sum; every invocation gets the result
#if USE_SUBGROUP_ADD
float block_sum(float v) {
    v = subgroupAdd(v);
    if (gl_NumSubgroups == 1) {
        return v;
    }
    if (subgroupElect()) {
        red[gl_SubgroupID] = v;
    }
    barrier();
    float r = 0.0f;
    for (uint i = 0; i < gl_NumSubgroups; ++i) {
        r += red[i];
    }
    barrier();
    return r;
}
#else
float block_sum(float v) {
    const uint tid = gl_LocalInvocationID.x;
    red[tid] = v;
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            red[tid] += red[tid + s];
        }
        barrier();
    }
    const float r = red[0];
    barrier();
    return r;
}
#endif

float sigmoid_f(float x) {
    return 1.0f/(1.0f + exp(-x));
}
