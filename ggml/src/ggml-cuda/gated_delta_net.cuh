#include "common.cuh"
#include "ggml.h"

// fused-kernel recurrent-state output; strides in elements (per-seq stride is always D, set in-kernel)
struct ggml_cuda_gated_delta_net_fused_cache {
    float * data;        // rollback slot 0
    int64_t slot_stride; // between rollback slots (0 when K==1)
};

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// fused-kernel recurrent-state input: the state of sequence s is row ids[s] of the cache at data (the GET_ROWS that
// gathers it is skipped, see ggml_cuda_match_gdn_state_rows)
struct ggml_cuda_gated_delta_net_state_rows {
    const float   * data;
    const int32_t * ids;
    int64_t         row_stride; // in floats
};

// same op, but writes the snapshot(s) into the cache instead of dst (see ggml_cuda_try_gdn_cache_fusion)
void ggml_cuda_op_gated_delta_net_fused_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                              ggml_cuda_gated_delta_net_fused_cache cache);

// same op with the state output and/or input fused (either may be nullptr)
void ggml_cuda_op_gated_delta_net_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst,
                                        const ggml_cuda_gated_delta_net_fused_cache * cache,
                                        const ggml_cuda_gated_delta_net_state_rows *  state_rows);
