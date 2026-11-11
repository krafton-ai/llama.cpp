#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_8bit_storage : require

#if USE_SUBGROUP_ADD || USE_SUBGROUP_ADD_NO_SHMEM
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#endif

#ifdef MUL_MAT_ID
#define EXPERT_COUNT 8
#endif

#include "mul_mat_vec_iface.glsl"

#ifndef MUL_MAT_ID
// MAT_VEC_FUSION_FLAGS_HC_POST: comb (post is fuse1)
layout (binding = 5) readonly buffer Fuse2 {D_TYPE data_fuse2[];};
#endif

layout (push_constant) uniform parameter
{
    uint ncols;
    uint stride_a;
    uint stride_b;
    uint stride_d;

    uint batch_stride_a;
    uint batch_stride_b;
    uint batch_stride_d;

    uint fusion_flags;

#ifdef MUL_MAT_ID
    uint nei0;
    uint ne11;
    uint expert_i1;
    uint nbi1;
#else
    uint base_work_group_y;
    uint ne02;
    uint ne12;
    uint broadcast2;
    uint broadcast3;
    uint fuse1_off0;   // MAT_VEC_FUSION_FLAGS_HC_POST: element offsets of post in fuse1 and of comb in fuse2
    uint fuse1_off1;
#endif
} p;

#ifdef MUL_MAT_ID
uint expert_id;
#endif

void get_offsets(out uint a_offset, out uint b_offset, out uint d_offset) {
#ifdef MUL_MAT_ID
    const uint expert_i0 = gl_WorkGroupID.y;
#else
    const uint batch_idx = gl_WorkGroupID.y + p.base_work_group_y;
#endif

#ifndef MUL_MAT_ID
    uint batch_idx_a = 0;
    if (batch_idx != 0) {
        const uint i13 = batch_idx / p.ne12;
        const uint i12 = batch_idx % p.ne12;

        const uint i03 = i13 / p.broadcast3;
        const uint i02 = i12 / p.broadcast2;

        batch_idx_a = i03 * p.ne02 + i02;
    }
#else
    expert_id = data_ids[expert_i0 + p.expert_i1 * p.nbi1];
#endif

    a_offset =
#ifdef MUL_MAT_ID
            expert_id * (p.batch_stride_a / QUANT_K);
#else
            batch_idx_a * (p.batch_stride_a / QUANT_K);
#endif
    b_offset =
#ifdef MUL_MAT_ID
            (expert_i0 % p.ne11) * p.stride_b + p.expert_i1 * p.batch_stride_b;
#else
            batch_idx * p.batch_stride_b;
#endif
    d_offset =
#ifdef MUL_MAT_ID
            expert_i0 * p.stride_d + p.expert_i1 * p.batch_stride_d;
#else
            batch_idx * p.batch_stride_d;
#endif
}

layout (constant_id = 0) const uint BLOCK_SIZE = 32;
layout (constant_id = 1) const uint NUM_ROWS = 1;
layout (constant_id = 2) const uint NUM_COLS = 1;

// MAT_VEC_FUSION_FLAGS_GLU: rows n < num_rows/2 of the workgroup are gate rows, the rest the matching up rows
uint glu_row(const uint first_row, const uint n, const uint num_rows) {
#ifndef MUL_MAT_ID
    if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU) != 0) {
        const uint h = num_rows / 2;
        return n < h ? first_row + n : p.stride_d + first_row + n - h;
    }
#endif
    return first_row + n;
}

#ifndef MUL_MAT_ID
// MAT_VEC_FUSION_FLAGS_GEGLU_MUL, the same formula as geglu.comp
FLOAT_TYPE geglu_mul(const FLOAT_TYPE a, const uint i) {
    const float x = float(a);
    const float val = 0.79788456080286535587989211986876f*x*(1.0f + 0.044715f*x*x);
    return FLOAT_TYPE(0.5f*x*(2.0f - 2.0f / (exp(2 * val) + 1)) * float(data_fuse0[i]));
}

// MAT_VEC_FUSION_FLAGS_HC_POST(_SUM): the 4 (5) stream outputs of row i (p.stride_d = n rows); base = d_offset
void store_hc_post(const uint base, const uint i, const FLOAT_TYPE v) {
    const uint n = p.stride_d;
    float r[4];
    [[unroll]] for (uint s = 0; s < 4; ++s) {
        r[s] = float(data_fuse0[i + s*n]);
    }
    float tot = 0.0f;
    [[unroll]] for (uint d = 0; d < 4; ++d) {
        float o = float(v) * float(data_fuse1[p.fuse1_off0 + d]);
        [[unroll]] for (uint s = 0; s < 4; ++s) {
            o = fma(r[s], float(data_fuse2[p.fuse1_off1 + d + 4*s]), o);
        }
        data_d[base + i + d*n] = D_TYPE(o);
        tot += o;
    }
    if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_HC_POST_SUM) != 0) {
        data_d[base + i + 4*n] = D_TYPE(tot);
    }
}

// writes silu(gate) * up for the num_rows/2 outputs of the workgroup; value(j, n) is the summed row n
#define STORE_GLU(value) \
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) { \
        [[unroll]] for (uint n = 0; n < num_rows / 2; ++n) { \
            const FLOAT_TYPE g = value(j, n); \
            data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(g / (FLOAT_TYPE(1) + exp(-g)) * value(j, n + num_rows / 2)); \
        } \
    }
#endif

#ifdef USE_SUBGROUP_ADD_NO_SHMEM
void reduce_result(inout FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

#ifndef MUL_MAT_ID
#define TEMP_VALUE(j, n) temp[j][n]
    if (tid == 0 && (p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU) != 0) {
        STORE_GLU(TEMP_VALUE)
        return;
    }
#undef TEMP_VALUE
#endif
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GEGLU_MUL) != 0) {
                    temp[j][n] = geglu_mul(temp[j][n], j*p.batch_stride_d + d_offset + first_row + n);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_HC_POST) != 0) {
                    store_hc_post(d_offset, first_row + n, temp[j][n]);
                    continue;
                }
#endif
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
            }
        }
    }
}
#else
shared FLOAT_TYPE tmpsh[NUM_COLS][NUM_ROWS][BLOCK_SIZE];

void reduce_result(FLOAT_TYPE temp[NUM_COLS][NUM_ROWS], const in uint32_t d_offset, const in uint32_t first_row, const in uint32_t num_rows, const in uint32_t tid) {
    // subgroupAdd is probably faster on devices that support it,
    // particularly when the workgroup has more than one subgroup
#if USE_SUBGROUP_ADD
    // sum up partial sums within a subgroup
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            temp[j][n] = subgroupAdd(temp[j][n]);
        }
    }

    // Go through shared memory to sum partials across subgroups
    if (gl_SubgroupInvocationID == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                tmpsh[j][n][gl_SubgroupID] = temp[j][n];
            }
        }
    }
    barrier();
#ifndef MUL_MAT_ID
    if (tid == 0 && (p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU) != 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
            }
        }
#define TEMP_VALUE(j, n) temp[j][n]
        STORE_GLU(TEMP_VALUE)
#undef TEMP_VALUE
        return;
    }
#endif
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                temp[j][n] = FLOAT_TYPE(0);
                [[unroll]] for (uint s = 0; s < gl_NumSubgroups; ++s) {
                    temp[j][n] += tmpsh[j][n][s];
                }
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    temp[j][n] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    temp[j][n] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GEGLU_MUL) != 0) {
                    temp[j][n] = geglu_mul(temp[j][n], j*p.batch_stride_d + d_offset + first_row + n);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_HC_POST) != 0) {
                    store_hc_post(d_offset, first_row + n, temp[j][n]);
                    continue;
                }
#endif
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(temp[j][n]);
            }
        }
    }
#else
    // sum up partial sums and write back result
    [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
        [[unroll]] for (uint n = 0; n < num_rows; ++n) {
            tmpsh[j][n][tid] = temp[j][n];
        }
    }
    barrier();
    [[unroll]] for (uint s = BLOCK_SIZE/2; s > 0; s >>= 1) {
        if (tid < s) {
            [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
                [[unroll]] for (uint n = 0; n < num_rows; ++n) {
                    tmpsh[j][n][tid] += tmpsh[j][n][tid + s];
                }
            }
        }
        barrier();
    }
#ifndef MUL_MAT_ID
#define TMPSH_VALUE(j, n) tmpsh[j][n][0]
    if (tid == 0 && (p.fusion_flags & MAT_VEC_FUSION_FLAGS_GLU) != 0) {
        STORE_GLU(TMPSH_VALUE)
        return;
    }
#undef TMPSH_VALUE
#endif
    if (tid == 0) {
        [[unroll]] for (uint j = 0; j < NUM_COLS; ++j) {
            [[unroll]] for (uint n = 0; n < num_rows; ++n) {
#ifdef MUL_MAT_ID
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[expert_id*p.stride_d + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE0) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse0[expert_i0]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_SCALE1) != 0) {
                    const uint expert_i0 = gl_GlobalInvocationID.y;
                    tmpsh[j][n][0] *= FLOAT_TYPE(data_fuse1[expert_i0]);
                }
#else
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS0) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse0[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_BIAS1) != 0) {
                    tmpsh[j][n][0] += FLOAT_TYPE(data_fuse1[j*p.batch_stride_d + d_offset + first_row + n]);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_GEGLU_MUL) != 0) {
                    tmpsh[j][n][0] = geglu_mul(tmpsh[j][n][0], j*p.batch_stride_d + d_offset + first_row + n);
                }
                if ((p.fusion_flags & MAT_VEC_FUSION_FLAGS_HC_POST) != 0) {
                    store_hc_post(d_offset, first_row + n, tmpsh[j][n][0]);
                    continue;
                }
#endif
                data_d[j*p.batch_stride_d + d_offset + first_row + n] = D_TYPE(tmpsh[j][n][0]);
            }
        }
    }
#endif
}
#endif
