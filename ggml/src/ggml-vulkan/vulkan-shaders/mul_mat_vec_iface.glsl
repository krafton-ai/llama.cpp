#include "types.glsl"

#define MAT_VEC_FUSION_FLAGS_BIAS0 0x1
#define MAT_VEC_FUSION_FLAGS_BIAS1 0x2
#define MAT_VEC_FUSION_FLAGS_SCALE0 0x4
#define MAT_VEC_FUSION_FLAGS_SCALE1 0x8
// rows [0, n) and [n, 2n) of the matrix are the gate and up halves of a swiglu: the workgroup computes row r of each
// half and writes silu(gate) * up to output r (p.stride_d = n); the shader maps its rows with glu_row()
#define MAT_VEC_FUSION_FLAGS_GLU 0x10
// the output is geglu(row, fuse0) = gelu_tanh(row) * fuse0[row] (a split GEGLU with a second, external input)
#define MAT_VEC_FUSION_FLAGS_GEGLU_MUL 0x20
// the output feeds an mHC hc_post with 4 streams (dsv4_hc_post): fuse0 = the residual streams [n, 4], fuse1 = post (4)
// and fuse2 = comb (4 x 4); row i writes stream d: v*post[d] + sum_s res[i, s]*comb[d + 4s]
#define MAT_VEC_FUSION_FLAGS_HC_POST 0x40
// ... and the sum of the 4 streams of row i to the second output (the stream sum after the hc_post)
#define MAT_VEC_FUSION_FLAGS_HC_POST_SUM 0x80

layout (binding = 0) readonly buffer A {A_TYPE data_a[];};
#if defined(A_TYPEV4)
layout (binding = 0) readonly buffer AV4 {A_TYPEV4 data_a_v4[];};
#endif
#if defined(A_TYPE_PACKED16)
layout (binding = 0) readonly buffer A_PACKED16 {A_TYPE_PACKED16 data_a_packed16[];};
#endif
#if defined(A_TYPE_PACKED32)
layout (binding = 0) readonly buffer A_PACKED32 {A_TYPE_PACKED32 data_a_packed32[];};
#endif
#if defined(A_TYPE_PACKED64)
layout (binding = 0) readonly buffer A_PACKED64 {A_TYPE_PACKED64 data_a_packed64[];};
#endif

layout (binding = 1) readonly buffer B {B_TYPE data_b[];};
#ifdef B_TYPEV2
layout (binding = 1) readonly buffer BV2 {B_TYPEV2 data_b_v2[];};
#endif
#ifdef B_TYPEV4
layout (binding = 1) readonly buffer BV4 {B_TYPEV4 data_b_v4[];};
#endif

layout (binding = 2) writeonly buffer D {D_TYPE data_d[];};

layout (binding = 3) readonly buffer Fuse0 {D_TYPE data_fuse0[];};
layout (binding = 4) readonly buffer Fuse1 {D_TYPE data_fuse1[];};

#ifdef MUL_MAT_ID
layout (binding = 5) readonly buffer IDS {int data_ids[];};
#endif

