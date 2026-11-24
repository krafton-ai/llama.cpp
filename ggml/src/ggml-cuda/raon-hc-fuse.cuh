#pragma once
#include "common.cuh"

// Raon mHC and RDA graph fusions.
// Each function returns true when the pattern matched and the fused kernel was launched.
bool ggml_cuda_raon_hc_try_fuse(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i, int * n_skip);

// SiTU graph fusion with both projections scheduled before activation.
bool ggml_cuda_raon_situ_match(const ggml_cgraph * cgraph, int i);
void ggml_cuda_raon_situ_run(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
