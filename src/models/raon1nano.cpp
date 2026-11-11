#include "models.h"

#include "llama-impl.h"

#include "llama-kv-cache.h"
#include "llama-memory-recurrent.h"

void llama_model_raon1_nano::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS,    hparams.f_norm_rms_eps);
    ml.get_key(LLM_KV_ATTENTION_KEY_LENGTH_MLA,       hparams.n_embd_head_k_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_VALUE_LENGTH_MLA,     hparams.n_embd_head_v_mla_impl);
    ml.get_key(LLM_KV_ATTENTION_Q_LORA_RANK,          hparams.n_lora_q);
    ml.get_key(LLM_KV_ATTENTION_KV_LORA_RANK,         hparams.n_lora_kv);
    ml.get_key(LLM_KV_ATTENTION_LATENT_NORM_RMS_EPS,  hparams.f_norm_rms_eps_latent);
    ml.get_key(LLM_KV_SSM_CONV_KERNEL,                hparams.ssm_d_conv);
    ml.get_key(LLM_KV_SSM_TIME_STEP_RANK,             hparams.ssm_dt_rank);
    ml.get_key(LLM_KV_KDA_HEAD_DIM,                   hparams.n_embd_head_kda);
    ml.get_key(LLM_KV_KDA_GATE_LOWER_BOUND,           hparams.kda_gate_lower_bound);
    ml.get_key(LLM_KV_HYPER_CONNECTION_COUNT,         hparams.dsv4_hc_mult);
    ml.get_key(LLM_KV_HYPER_CONNECTION_EPSILON,       hparams.dsv4_hc_eps);
    ml.get_key(LLM_KV_HYPER_CONNECTION_MAGNITUDE,     hparams.hc_magnitude);
    ml.get_key(LLM_KV_EMBEDDING_LENGTH_PER_LAYER,     hparams.n_embd_per_layer);

    // dense FFN in every layer
    hparams.n_layer_dense_lead = hparams.n_layer();

    auto require = [](bool ok, const std::string & what) {
        if (!ok) {
            throw std::runtime_error("invalid Raon-1 nano hyperparameters: " + what);
        }
    };

    // RDA layers have head_count_kv 0, MLA layers one latent KV head
    uint32_t n_rda = 0;
    for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
        hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
        n_rda += hparams.is_recr_impl[i];
        require(hparams.is_recr_impl[i] || hparams.n_head_kv(i) == 1, format("MLA layer %u has %u KV heads, not 1", i, hparams.n_head_kv(i)));
        // the recurrent state size (n_embd_r, n_embd_s) is taken from n_head(), the head count of layer 0
        require(!hparams.is_recr_impl[i] || hparams.n_head(i) == hparams.n_head(), format("RDA layer %u has %u heads, layer 0 has %u", i, hparams.n_head(i), hparams.n_head()));
    }
    require(n_rda > 0 && n_rda < hparams.n_layer(), format("needs both RDA and MLA layers, got %u RDA of %u layers", n_rda, hparams.n_layer()));
    require(hparams.is_recr_impl[0], "layer 0 is not an RDA layer");

    // only 4 mHC streams and a conv width of 4 have been run (the released model)
    require(hparams.dsv4_hc_mult >= 2 && hparams.dsv4_hc_mult <= 8, format("hyper_connection.count %u is not in [2, 8]", hparams.dsv4_hc_mult));
    require(hparams.ssm_d_conv == 4, format("ssm.conv_kernel %u is not 4", hparams.ssm_d_conv));
    require(hparams.ssm_dt_rank > 0, "ssm.time_step_rank is 0");
    require(hparams.n_lora_q > 0 && hparams.n_lora_kv > 0, "q_lora_rank or kv_lora_rank is 0");
    // the absorbed MLA attends over the latent: V is the kv_lora_rank-wide latent
    require(hparams.n_embd_head_v_full == hparams.n_lora_kv, format("the MLA value length %u is not kv_lora_rank %u", hparams.n_embd_head_v_full, hparams.n_lora_kv));
    require(hparams.n_embd_head_k_mla() > hparams.n_rot(), format("the MLA key length %u is not larger than the RoPE dimension %u", hparams.n_embd_head_k_mla(), hparams.n_rot()));
    require(hparams.f_norm_rms_eps_latent > 0 && hparams.dsv4_hc_eps > 0, "a normalization epsilon is not positive");
    require(hparams.hc_magnitude > 0, "hyper_connection.magnitude is not positive");
    require(hparams.n_embd_per_layer > 0, "embedding_length_per_layer_input is 0");
    require(hparams.kda_gate_lower_bound < 0, "kda.gate_lower_bound is not negative");
    require(hparams.n_embd_head_kda == 16 || hparams.n_embd_head_kda == 32 || hparams.n_embd_head_kda == 64 || hparams.n_embd_head_kda == 128,
            format("RDA head dimension %u is not 16, 32, 64 or 128", hparams.n_embd_head_kda));

    // the extracted layer inputs are the full mHC residual: hc streams of n_embd each
    hparams.n_embd_layer_inp_impl = hparams.dsv4_hc_mult * hparams.n_embd;

    type = LLM_TYPE_UNKNOWN;
}

void llama_model_raon1_nano::load_arch_tensors(llama_model_loader & ml) {
    LLAMA_LOAD_LOCALS;

    const auto nf = nextn_flags(ml, LLM_TENSOR_HC_ATTN_FN);

    const int64_t hc = hparams.dsv4_hc_mult;
    // selector width per mHC site: pre (hc) + post (hc) + combination ((hc-1)^2, Helmert coordinates)
    const int64_t hc_mix_dim = 2*hc + (hc - 1)*(hc - 1);
    // chart = [K {hc_mix_dim, hc*hc} | r {hc*hc} | c0 {hc, hc}], see build_hc_pre
    const int64_t hc_chart_dim = hc_mix_dim*hc*hc + 2*hc*hc;
    const int64_t n_ple = hparams.n_embd_per_layer;

    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, 0);

    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, nf.trunk);
    output      = create_tensor(tn(LLM_TENSOR_OUTPUT,      "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    // tied head
    if (output == nullptr) {
        output = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD, "weight"), {n_embd, n_vocab}, TENSOR_DUPLICATED);
    }

    // the projection and its norm are repeating-layer tensors: named without, but created with, a layer number (as gemma3n)
    per_layer_tok_embd   = create_tensor(tn(LLM_TENSOR_PER_LAYER_TOKEN_EMBD, "weight"), {n_ple * n_layer, n_vocab}, nf.trunk);
    per_layer_model_proj = create_tensor(tn(LLM_TENSOR_PER_LAYER_MODEL_PROJ, "weight", 0), {n_embd, n_ple * n_layer}, nf.trunk);
    per_layer_proj_norm  = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ_NORM,  "weight", 0), {n_ple}, nf.trunk);

    // the MTP block (layers n_layer..n_layer_all) is an MLA layer without the per-layer input
    for (int i = 0; i < n_layer_all; ++i) {
        auto & layer = layers[i];
        const bool mtp = i >= n_layer;
        const int flags = mtp ? nf.mtp : nf.trunk;

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, flags);
        layer.ffn_norm  = create_tensor(tn(LLM_TENSOR_FFN_NORM,  "weight", i), {n_embd}, flags);

        layer.hc_attn_fn    = create_tensor(tn(LLM_TENSOR_HC_ATTN_FN,    "weight", i), {hc*n_embd, hc_mix_dim}, flags);
        layer.hc_attn_base  = create_tensor(tn(LLM_TENSOR_HC_ATTN_BASE,  "weight", i), {hc_mix_dim}, flags);
        layer.hc_attn_chart = create_tensor(tn(LLM_TENSOR_HC_ATTN_CHART, "weight", i), {hc_chart_dim}, flags);
        layer.hc_ffn_fn     = create_tensor(tn(LLM_TENSOR_HC_FFN_FN,     "weight", i), {hc*n_embd, hc_mix_dim}, flags);
        layer.hc_ffn_base   = create_tensor(tn(LLM_TENSOR_HC_FFN_BASE,   "weight", i), {hc_mix_dim}, flags);
        layer.hc_ffn_chart  = create_tensor(tn(LLM_TENSOR_HC_FFN_CHART,  "weight", i), {hc_chart_dim}, flags);

        const int64_t n_head_l = hparams.n_head(i);

        if (hparams.is_recr(i)) {
            const int64_t head_dim = hparams.n_embd_head_kda;
            const int64_t d_inner  = head_dim * n_head_l;

            // q | k | v | g in one projection, q | k | v in one depthwise conv; ssm_f_a rows are [f_a | b | b + rda];
            // exp(A_log) is folded into ssm_f_b and the bias, so the decay gate is lower_bound * sigmoid(f_b(f_a(x)) + dt_bias)
            layer.wqkv       = create_tensor(tn(LLM_TENSOR_ATTN_QKV,   "weight", i), {n_embd, 4*d_inner}, flags);
            layer.ssm_conv1d = create_tensor(tn(LLM_TENSOR_SSM_CONV1D, "weight", i), {hparams.ssm_d_conv, 3*d_inner}, flags);
            layer.ssm_f_a    = create_tensor(tn(LLM_TENSOR_SSM_F_A,    "weight", i), {n_embd, hparams.ssm_dt_rank + 2*n_head_l}, flags);
            layer.ssm_f_b    = create_tensor(tn(LLM_TENSOR_SSM_F_B,    "weight", i), {hparams.ssm_dt_rank, d_inner}, flags);
            layer.ssm_dt_b   = create_tensor(tn(LLM_TENSOR_SSM_DT,     "bias",   i), {d_inner}, flags);
            layer.ssm_o_norm = create_tensor(tn(LLM_TENSOR_SSM_NORM,   "weight", i), {head_dim}, flags);
            layer.wo         = create_tensor(tn(LLM_TENSOR_ATTN_OUT,   "weight", i), {d_inner, n_embd}, flags);
        } else {
            const int64_t q_lora_rank      = hparams.n_lora_q;
            const int64_t kv_lora_rank     = hparams.n_lora_kv;
            const int64_t n_embd_head_k    = hparams.n_embd_head_k_mla();
            const int64_t n_embd_head_v    = hparams.n_embd_head_v_mla();
            const int64_t qk_rope_head_dim = hparams.n_rot();
            const int64_t qk_nope_head_dim = n_embd_head_k - qk_rope_head_dim;

            layer.attn_q_a_norm  = create_tensor(tn(LLM_TENSOR_ATTN_Q_A_NORM,  "weight", i), {q_lora_rank}, flags);
            layer.attn_kv_a_norm = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_NORM, "weight", i), {kv_lora_rank}, flags);
            layer.wq_a      = create_tensor(tn(LLM_TENSOR_ATTN_Q_A,      "weight", i), {n_embd, q_lora_rank}, flags);
            layer.wq_b      = create_tensor(tn(LLM_TENSOR_ATTN_Q_B,      "weight", i), {q_lora_rank, n_head_l*n_embd_head_k}, flags);
            layer.wkv_a_mqa = create_tensor(tn(LLM_TENSOR_ATTN_KV_A_MQA, "weight", i), {n_embd, kv_lora_rank + qk_rope_head_dim}, flags);
            // some files also carry an expanded kv_b; it is not used
            if (ml.get_tensor_meta(tn(LLM_TENSOR_ATTN_KV_B, "weight", i).str().c_str())) {
                create_tensor(tn(LLM_TENSOR_ATTN_KV_B, "weight", i),
                        {kv_lora_rank + qk_rope_head_dim, n_head_l*(n_embd_head_k + n_embd_head_v)}, flags | TENSOR_SKIP);
            }
            layer.wk_b      = create_tensor(tn(LLM_TENSOR_ATTN_K_B,  "weight", i), {qk_nope_head_dim, kv_lora_rank, n_head_l}, flags);
            layer.wv_b      = create_tensor(tn(LLM_TENSOR_ATTN_V_B,  "weight", i), {kv_lora_rank, n_embd_head_v, n_head_l}, flags);
            layer.wqkv_gate = create_tensor(tn(LLM_TENSOR_ATTN_GATE, "weight", i), {n_embd, n_head_l*n_embd_head_v}, flags);
            layer.wo        = create_tensor(tn(LLM_TENSOR_ATTN_OUT,  "weight", i), {n_head_l*n_embd_head_v, n_embd}, flags);
        }

        if (!mtp) {
            layer.per_layer_inp_gate  = create_tensor(tn(LLM_TENSOR_PER_LAYER_INP_GATE,  "weight", i), {n_embd, n_ple}, flags);
            layer.per_layer_proj      = create_tensor(tn(LLM_TENSOR_PER_LAYER_PROJ,      "weight", i), {n_ple, n_embd}, flags);
            layer.per_layer_post_norm = create_tensor(tn(LLM_TENSOR_PER_LAYER_POST_NORM, "weight", i), {n_embd}, flags);
        }

        // ffn_up rows are [gate | up], for one swiglu
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, 2*n_ff}, flags);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff, n_embd}, flags);

        if (mtp) {
            layer.nextn.eh_proj          = create_tensor(tn(LLM_TENSOR_NEXTN_EH_PROJ,          "weight", i), {2*n_embd, n_embd}, flags);
            layer.nextn.enorm            = create_tensor(tn(LLM_TENSOR_NEXTN_ENORM,            "weight", i), {n_embd}, flags);
            layer.nextn.hnorm            = create_tensor(tn(LLM_TENSOR_NEXTN_HNORM,            "weight", i), {n_embd}, flags);
            layer.nextn.shared_head_norm = create_tensor(tn(LLM_TENSOR_NEXTN_SHARED_HEAD_NORM, "weight", i), {n_embd}, flags);
        }
    }
}

std::unique_ptr<llm_graph_context> llama_model_raon1_nano::build_arch_graph(const llm_graph_params & params) const {
    if (params.gtype == LLM_GRAPH_TYPE_DECODER_MTP) {
        return std::make_unique<graph_mtp>(*this, params);
    }
    return std::make_unique<graph>(*this, params);
}

// mHC selector on the folded layout (conversion/raon1_nano.py): fn rows are pre-multiplied by alpha,
// chart holds K = gamma * (H (x) H) acting on the whole mix vector (zero on pre/post), r = -1 / (gamma * anchor)
// and c0 = (1 - gamma) I + gamma * anchor. With e' = K mixes = gamma * e:
// rho = 1 + max(-e / anchor) and comb = I + gamma * (anchor + e / rho - I) = c0 + e' / rho.
// max(-e / anchor) >= 0 holds without a clamp: every row of e = H z H^T sums to zero.
ggml_tensor * llama_model_raon1_nano::graph::build_hc_pre(
        ggml_tensor * x, ggml_tensor * fn, ggml_tensor * base, ggml_tensor * chart,
        ggml_tensor * norm_w, ggml_tensor ** post, ggml_tensor ** comb, int il) {
    const int64_t hc   = hparams.dsv4_hc_mult;
    const int64_t hc2  = hc*hc;
    const int64_t nt   = x->ne[2];
    const int64_t nmix = fn->ne[1];

    ggml_tensor * flat = ggml_reshape_2d(ctx0, x, n_embd*hc, nt);

    // fn (x / rms(x)), the same as (fn x) / rms(x)
    ggml_tensor * mixes = build_lora_mm(fn, ggml_rms_norm(ctx0, flat, hparams.dsv4_hc_eps));
    cb(mixes, "hc_mixes", il);
    mixes = ggml_add(ctx0, mixes, base);

    // one sigmoid for pre and post
    ggml_tensor * sg  = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, mixes, 2*hc, nt, mixes->nb[1], 0));
    ggml_tensor * pre = ggml_view_2d(ctx0, sg, hc, nt, sg->nb[1], 0);
    *post = ggml_view_2d(ctx0, sg, hc, nt, sg->nb[1], hc*sg->nb[0]);

    ggml_tensor * K  = ggml_view_2d(ctx0, chart, nmix, hc2, nmix*chart->nb[0], 0);
    ggml_tensor * r  = ggml_view_1d(ctx0, chart, hc2, nmix*hc2*chart->nb[0]);
    ggml_tensor * c0 = ggml_view_2d(ctx0, chart, hc, hc, hc*chart->nb[0], (nmix*hc2 + hc2)*chart->nb[0]);

    ggml_tensor * e   = ggml_mul_mat(ctx0, K, mixes); // [hc*hc, nt] = gamma * e
    ggml_tensor * rho = ggml_pool_1d(ctx0, ggml_mul(ctx0, e, r), GGML_OP_POOL_MAX, hc2, hc2, 0);
    rho = ggml_reshape_3d(ctx0, ggml_scale_bias(ctx0, rho, 1.0f, 1.0f), 1, 1, nt);
    *comb = ggml_add(ctx0, ggml_div(ctx0, ggml_reshape_3d(ctx0, e, hc, hc, nt), rho), c0);
    if (hparams.hc_magnitude != 1.0f) {
        *post = ggml_scale(ctx0, *post, hparams.hc_magnitude);
    }
    cb(*comb, "hc_comb", il);
    cb(pre, "hc_pre", il);
    cb(*post, "hc_post", il);

    ggml_tensor * result = nullptr;
    if (cparams.fused_dsv4_hc_pre) {
        result = ggml_dsv4_hc_pre(ctx0, x, pre);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_PRE, result, il});
    } else {
        for (int64_t i = 0; i < hc; ++i) {
            ggml_tensor * xi = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], i*x->nb[1]);
            ggml_tensor * wi = ggml_view_2d(ctx0, pre, 1, nt, pre->nb[1], i*pre->nb[0]);
            ggml_tensor * term = ggml_mul(ctx0, xi, wi);
            result = result ? ggml_add(ctx0, result, term) : term;
        }
    }
    return build_norm(result, norm_w, nullptr, LLM_NORM_RMS, il);
}

ggml_tensor * llama_model_raon1_nano::graph::build_hc_post(
        ggml_tensor * x, ggml_tensor * branch, ggml_tensor * post, ggml_tensor * comb, int il) {
    if (cparams.fused_dsv4_hc_post) {
        ggml_tensor * result = ggml_dsv4_hc_post(ctx0, branch, x, post, comb);
        res->add_fused_node({LLM_FUSED_OP_DSV4_HC_POST, result, il});
        return result;
    }

    const int64_t hc = hparams.dsv4_hc_mult;
    const int64_t nt = x->ne[2];
    ggml_tensor * result = nullptr;
    for (int64_t dst = 0; dst < hc; ++dst) {
        ggml_tensor * sum = nullptr;
        for (int64_t src = 0; src < hc; ++src) {
            ggml_tensor * xi = ggml_view_2d(ctx0, x, n_embd, nt, x->nb[2], src*x->nb[1]);
            ggml_tensor * wi = ggml_view_2d(ctx0, comb, 1, nt, comb->nb[2], dst*comb->nb[0] + src*comb->nb[1]);
            ggml_tensor * term = ggml_mul(ctx0, xi, wi);
            sum = sum ? ggml_add(ctx0, sum, term) : term;
        }
        ggml_tensor * wi = ggml_view_2d(ctx0, post, 1, nt, post->nb[1], dst*post->nb[0]);
        sum = ggml_add(ctx0, sum, ggml_mul(ctx0, branch, wi));
        sum = ggml_reshape_3d(ctx0, sum, n_embd, 1, nt);
        result = result ? ggml_concat(ctx0, result, sum, 1) : sum;
    }
    return result;
}

// Raon delta attention: a gated delta-rule layer whose write gate differs from its erase gate.
// The write/erase ratio exp(softplus(-b) - softplus(-b-d)) = sigmoid(b + d) / sigmoid(b) scales V,
// from the [b | b + d] rows of ssm_f_a
ggml_tensor * llama_model_raon1_nano::graph::build_rda_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_rs * inp_rs, int il) {
    const int64_t n_head       = hparams.n_head(il);
    const int64_t head_dim     = hparams.n_embd_head_kda;
    const int64_t d_inner      = head_dim * n_head;
    const int64_t n_seq_tokens = ubatch.n_seq_tokens;
    const int64_t n_seqs       = ubatch.n_seqs;

    // wqkv rows: q | k | v | g (the output gate); the conv takes q | k | v
    ggml_tensor * qkvg = build_lora_mm(layer.wqkv, cur);
    ggml_tensor * qkv  = ggml_view_3d(ctx0, qkvg, 3*d_inner, n_seq_tokens, n_seqs,
            qkvg->nb[1], qkvg->nb[1]*n_seq_tokens, 0);
    ggml_tensor * conv_input = build_conv_state(inp_rs, inp_rs->mctx->get_r_l(il), qkv, hparams.ssm_d_conv, 3*d_inner, il);
    ggml_tensor * conv = ggml_silu(ctx0, ggml_ssm_conv(ctx0, conv_input, layer.ssm_conv1d));
    cb(conv, "rda_conv", il);

    const size_t nb1 = ggml_row_size(conv->type, 3*d_inner);
    // q and k heads are adjacent: one L2 norm, same as HF F.normalize(eps=1e-6): x / max(||x||, eps)
    ggml_tensor * qk = ggml_l2_norm(ctx0, ggml_view_4d(ctx0, conv, head_dim, 2*n_head, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_dim), nb1, nb1*n_seq_tokens, 0), 1e-6f);
    ggml_tensor * Qcur = ggml_view_4d(ctx0, qk, head_dim, n_head, n_seq_tokens, n_seqs, qk->nb[1], qk->nb[2], qk->nb[3], 0);
    ggml_tensor * Kcur = ggml_view_4d(ctx0, qk, head_dim, n_head, n_seq_tokens, n_seqs, qk->nb[1], qk->nb[2], qk->nb[3], n_head*qk->nb[1]);
    ggml_tensor * Vcur = ggml_view_4d(ctx0, conv, head_dim, n_head, n_seq_tokens, n_seqs,
            ggml_row_size(conv->type, head_dim), nb1, nb1*n_seq_tokens, 2*ggml_row_size(conv->type, d_inner));

    // ssm_f_a rows: [f_a (decay rank) | b | b + rda]
    const int64_t rank = hparams.ssm_dt_rank;
    ggml_tensor * fab = build_lora_mm(layer.ssm_f_a, cur);
    ggml_tensor * f_a = ggml_view_2d(ctx0, fab, rank, n_tokens, fab->nb[1], 0);

    ggml_tensor * g1 = ggml_add(ctx0, build_lora_mm(layer.ssm_f_b, f_a), layer.ssm_dt_b);
    g1 = ggml_scale(ctx0, ggml_sigmoid(ctx0, g1), hparams.kda_gate_lower_bound);
    g1 = ggml_reshape_4d(ctx0, g1, head_dim, n_head, n_seq_tokens, n_seqs);

    ggml_tensor * beta  = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, fab, n_head, n_tokens, fab->nb[1], rank*fab->nb[0]));
    ggml_tensor * write = ggml_sigmoid(ctx0, ggml_view_2d(ctx0, fab, n_head, n_tokens, fab->nb[1], (rank + n_head)*fab->nb[0]));
    ggml_tensor * ratio = ggml_reshape_4d(ctx0, ggml_div(ctx0, write, beta), 1, n_head, n_seq_tokens, n_seqs);
    Vcur = ggml_mul(ctx0, Vcur, ratio);
    beta = ggml_reshape_4d(ctx0, beta, 1, n_head, n_seq_tokens, n_seqs);
    cb(g1, "rda_g1", il);
    cb(Vcur, "rda_v_scaled", il);
    cb(beta, "rda_beta", il);

    ggml_tensor * g2 = ggml_view_3d(ctx0, qkvg, head_dim, n_head, n_tokens,
            ggml_row_size(qkvg->type, head_dim), qkvg->nb[1], 3*ggml_row_size(qkvg->type, d_inner));

    ggml_tensor * ssm_states_all = inp_rs->mctx->get_s_l(il);
    ggml_tensor * state = build_rs(inp_rs, ssm_states_all, hparams.n_embd_s(), n_seqs);
    state = ggml_reshape_4d(ctx0, state, head_dim, head_dim, n_head, n_seqs);
    ggml_tensor * output = build_recurrent_attn(inp_rs, ssm_states_all, Qcur, Kcur, Vcur, g1, beta, state, il);
    output = ggml_is_contiguous(output) ? output : ggml_cont(ctx0, output);
    cb(output, "rda_scan_out", il);

    // RMSNorm(o) * sigmoid(g2)
    ggml_tensor * o = ggml_reshape_3d(ctx0, output, head_dim, n_head, n_tokens);
    ggml_tensor * gated = ggml_mul(ctx0, build_norm(o, layer.ssm_o_norm, nullptr, LLM_NORM_RMS, il), ggml_sigmoid(ctx0, g2));

    cur = build_lora_mm(layer.wo, ggml_reshape_2d(ctx0, gated, d_inner, n_tokens));
    cb(cur, "rda_out", il);

    return cur;
}

ggml_tensor * llama_model_raon1_nano::graph::build_mla_layer(
        ggml_tensor * cur, const llama_layer & layer, llm_graph_input_attn_k * inp_attn_k, int il) {
    const int64_t n_head = hparams.n_head(il);
    const int64_t nr = hparams.n_rot();
    const int64_t nk = hparams.n_embd_head_k_mla();
    const int64_t nn = nk - nr;
    const int64_t rank = hparams.n_lora_kv;
    const float eps = hparams.f_norm_rms_eps_latent;
    ggml_tensor * q = build_lora_mm(layer.wq_a, cur);
    q = ggml_mul(ctx0, ggml_rms_norm(ctx0, q, eps), layer.attn_q_a_norm);
    q = build_lora_mm(layer.wq_b, q);
    ggml_tensor * kv = build_lora_mm(layer.wkv_a_mqa, cur);
    ggml_tensor * latent = ggml_view_2d(ctx0, kv, rank, n_tokens, kv->nb[1], 0);
    latent = ggml_mul(ctx0, ggml_rms_norm(ctx0, latent, eps), layer.attn_kv_a_norm);
    // NoPE: the "rope" dims of q and k are concatenated as they are, without rotation
    ggml_tensor * extra = ggml_view_3d(ctx0, kv, nr, 1, n_tokens, kv->nb[1], kv->nb[1], rank*kv->nb[0]);
    if (n_tokens > LLAMA_MLA_EXPAND_MIN_TOKENS && ubatch.n_seqs == 1 && cparams.flash_attn && cparams.mla_expand) {
        // long prefill: attend over per-head K/V expanded from the latent cache, a group of g heads at a time.
        // g is fixed from the context size (same graph for every ubatch): the most heads, a power of two, whose
        // F16 K/V of a full cache stay within 96 MiB
        const int64_t nv  = hparams.n_embd_head_v_mla();
        const int64_t per_head = (nk + nv)*(int64_t) cparams.n_ctx*(int64_t) sizeof(ggml_fp16_t);
        int64_t g = 1;
        while (2*g <= n_head && n_head % (2*g) == 0 && 2*g*per_head <= (int64_t) 96*1024*1024) {
            g *= 2;
        }
        const auto * mctx_k = inp_attn_k->mctx;
        ggml_tensor * kc = mctx_k->get_k(ctx0, il);                           // [rank + nr, 1, n_kv, 1]
        if (kc->type == GGML_TYPE_F16 && kc->ne[1] == 1 && kc->ne[3] == 1) {
            ggml_tensor * k_cur = ggml_concat(ctx0, ggml_reshape_3d(ctx0, latent, rank, 1, n_tokens), extra, 0);
            ggml_build_forward_expand(gf, q);
            ggml_build_forward_expand(gf, mctx_k->cpy_k(ctx0, k_cur, inp_attn_k->get_k_idxs(), il));
            const int64_t n_kv = kc->ne[2];
            ggml_tensor * kc2 = ggml_view_2d(ctx0, kc, rank + nr, n_kv, kc->nb[2], 0);
            ggml_tensor * out = nullptr;
            for (int64_t h0 = 0; h0 < n_head; h0 += g) {
                ggml_tensor * kvg = ggml_mla_expand_kv(ctx0, kc2, layer.wk_b, layer.wv_b, (int32_t) h0, (int32_t) g);
                res->add_fused_node({LLM_FUSED_OP_MLA_EXPAND_KV, kvg, il});
                const size_t e16 = ggml_element_size(kvg);
                ggml_tensor * kg = ggml_view_3d(ctx0, kvg, nk, g, n_kv, nk*e16, nk*g*e16, 0);
                ggml_tensor * vg = ggml_view_3d(ctx0, kvg, nv, g, n_kv, nv*e16, nv*g*e16, nk*g*n_kv*e16);
                ggml_tensor * qg = ggml_view_3d(ctx0, q, nk, g, n_tokens, nk*q->nb[0], q->nb[1], h0*nk*q->nb[0]);
                ggml_tensor * og = build_attn_mha(qg, kg, vg, nullptr, inp_attn_k->get_kq_mask(), nullptr, nullptr, 0, 1.0f/sqrtf(nk), il);
                out = out ? ggml_concat(ctx0, out, og, 0) : og;
            }
            cb(out, "mla_expanded", il);
            out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, cur)));
            cb(out, "mla_gated", il);
            return build_lora_mm(layer.wo, out);
        }
    }
    ggml_tensor * q_nope = ggml_view_3d(ctx0, q, nn, n_head, n_tokens, nk*q->nb[0], q->nb[1], 0);
    ggml_tensor * q_extra = ggml_view_3d(ctx0, q, nr, n_head, n_tokens, nk*q->nb[0], q->nb[1], nn*q->nb[0]);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q_nope = build_lora_mm(layer.wk_b, q_nope);
    q_nope = ggml_permute(ctx0, q_nope, 0, 2, 1, 3);
    q = ggml_concat(ctx0, q_nope, q_extra, 0);
    latent = ggml_reshape_3d(ctx0, latent, rank, 1, n_tokens);
    ggml_tensor * k = ggml_concat(ctx0, latent, extra, 0);
    ggml_tensor * out = build_attn(inp_attn_k, nullptr, nullptr, nullptr, q, k, latent,
            nullptr, nullptr, layer.wv_b, 1.0f/sqrtf(nk), il);
    out = ggml_mul(ctx0, out, ggml_sigmoid(ctx0, build_lora_mm(layer.wqkv_gate, cur)));
    cb(out, "mla_gated", il);
    return build_lora_mm(layer.wo, out);
}

llama_model_raon1_nano::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_build_delta_net_base(params) {
    const int64_t hc    = hparams.dsv4_hc_mult;
    const int64_t n_ple = hparams.n_embd_per_layer;

    // sum of the hc streams: [n_embd, hc, n_tokens] -> [n_embd, n_tokens]
    auto stream_sum = [&](ggml_tensor * x) {
        ggml_tensor * sum = nullptr;
        for (int64_t i = 0; i < hc; ++i) {
            ggml_tensor * xi = ggml_view_2d(ctx0, x, n_embd, x->ne[2], x->nb[2], i*x->nb[1]);
            sum = sum ? ggml_add(ctx0, sum, xi) : xi;
        }
        return sum;
    };

    ggml_tensor * inp = build_inp_embd(model.tok_embd);

    // per-layer inputs: ple = (RMSNorm_w(embd @ model_proj) + table[token] * sqrt(n_ple)) / sqrt(2), as [n_ple, n_tokens, n_layer]
    ggml_tensor * rows;
    if (ubatch.token) {
        rows = ggml_get_rows(ctx0, model.per_layer_tok_embd, res->t_inp_tokens);
        // keep the table lookup at the start of the graph, like the token embeddings
        ggml_build_forward_order(gf, rows);
        rows = ggml_reshape_3d(ctx0, rows, n_ple, n_layer, n_tokens);
    } else {
        // embeddings input: use the row of token 0, as gemma3n
        rows = ggml_cast(ctx0, ggml_view_1d(ctx0, model.per_layer_tok_embd, n_ple*n_layer, 0), GGML_TYPE_F32);
        rows = ggml_reshape_3d(ctx0, rows, n_ple, n_layer, 1);
    }
    rows = ggml_scale(ctx0, rows, sqrtf((float) n_ple));
    ggml_tensor * proj = build_lora_mm(model.per_layer_model_proj, inp);
    proj = ggml_reshape_3d(ctx0, proj, n_ple, n_layer, n_tokens);
    proj = build_norm(proj, model.per_layer_proj_norm, nullptr, LLM_NORM_RMS, -1);
    ggml_tensor * inp_ple = ggml_scale(ctx0, ggml_add(ctx0, proj, rows), 1.0f/sqrtf(2.0f));
    inp_ple = ggml_cont(ctx0, ggml_permute(ctx0, inp_ple, 0, 2, 1, 3));
    cb(inp_ple, "inp_per_layer", -1);

    inp = ggml_reshape_3d(ctx0, inp, n_embd, 1, n_tokens);
    inp = ggml_repeat_4d(ctx0, inp, n_embd, hc, n_tokens, 1);
    auto * inputs = build_inp_mem_hybrid_k();
    auto * inp_rs = inputs->get_recr();
    auto * inp_attn = inputs->get_attn();
    auto * out_ids = build_inp_out_ids();
    GGML_ASSERT(ubatch.n_seqs != 0 && ubatch.equal_seqs());
    GGML_ASSERT(ubatch.n_tokens == ubatch.n_seq_tokens*ubatch.n_seqs);

    for (int il = 0; il < n_layer; ++il) {
        // [n_embd, hc, n_tokens]: the output of layer il-1, stream-major per token
        res->t_layer_inp[il] = inp;

        const auto & layer = model.layers[il];
        ggml_tensor * post;
        ggml_tensor * comb;
        ggml_tensor * cur = build_hc_pre(inp, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_chart, layer.attn_norm, &post, &comb, il);
        cb(cur, "attn_norm", il);
        if (hparams.is_recr(il)) {
            cur = build_rda_layer(cur, layer, inp_rs, il);
        } else {
            cur = build_mla_layer(cur, layer, inp_attn, il);
        }
        inp = build_hc_post(inp, cur, post, comb, il);
        cb(inp, "hc_attn_out", il);
        cur = build_hc_pre(inp, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_chart, layer.ffn_norm, &post, &comb, il);
        cb(cur, "ffn_norm", il);
        cur = build_ffn(cur, layer.ffn_up, nullptr, nullptr, nullptr, nullptr, nullptr,
                layer.ffn_down, nullptr, nullptr, nullptr, LLM_FFN_SWIGLU, LLM_FFN_SEQ, il);
        inp = build_hc_post(inp, cur, post, comb, il);

        // per-layer input: streams += RMSNorm(proj(gelu_tanh(gate(stream sum)) * ple_il)), on every stream;
        // the gate is pre-divided by hc in conversion, so it takes the sum rather than the mean
        ggml_tensor * ple_il = ggml_view_2d(ctx0, inp_ple, n_ple, n_tokens, inp_ple->nb[1], il*inp_ple->nb[2]);
        ggml_tensor * g = ggml_geglu_split(ctx0, build_lora_mm(layer.per_layer_inp_gate, stream_sum(inp)), ple_il);
        g = build_lora_mm(layer.per_layer_proj, g);
        g = build_norm(g, layer.per_layer_post_norm, nullptr, LLM_NORM_RMS, il);
        cb(g, "ple_out", il);
        inp = ggml_add(ctx0, inp, ggml_reshape_3d(ctx0, g, n_embd, 1, n_tokens));

        inp = build_cvec(inp, il);
        cb(inp, "l_out", il);
    }
    res->t_layer_inp[n_layer] = inp;

    ggml_tensor * cur = ggml_scale(ctx0, stream_sum(inp), 1.0f/hc);
    if (crop_before_nextn(out_ids)) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cur = build_norm(cur, model.output_norm, nullptr, LLM_NORM_RMS, -1);
    // the MTP head reads the final-normed hidden state (the lm_head input)
    cb(cur, "h_nextn", -1);
    res->t_h_nextn = cur;
    if (crop_after_nextn(out_ids)) {
        cur = ggml_get_rows(ctx0, cur, out_ids);
    }
    cb(cur, "result_norm", -1);
    res->t_embd = cur;
    cur = build_lora_mm(model.output, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}

// MTP block: h = the trunk's final-normed hidden at position i, e = the embedding of token i+1;
// x = eh_proj([enorm(e) ; hnorm(h)]) is copied to every mHC stream, runs one MLA layer (no per-layer input),
// and the stream mean goes through shared_head_norm to the tied head. That normed state is also the next draft step's h.
llama_model_raon1_nano::graph_mtp::graph_mtp(const llama_model & model, const llm_graph_params & params) :
    graph(params, no_build{}) {
    GGML_ASSERT(hparams.n_layer_nextn == 1 && "Raon-1 nano MTP has a single block");
    ASSERT_EMBD_OR_TOKEN(ubatch);

    const int64_t hc = hparams.dsv4_hc_mult;
    const int il = hparams.n_layer();
    const auto & layer = model.layers[il];
    GGML_ASSERT(!hparams.is_recr(il));
    GGML_ASSERT(layer.nextn.eh_proj && layer.nextn.enorm && layer.nextn.hnorm && layer.nextn.shared_head_norm &&
            "MTP block missing, load the model with MTP enabled");

    auto inp = std::make_unique<llm_graph_input_embd_h>(hparams.n_embd_inp(), hparams.n_embd);
    inp->tokens = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_tokens);
    ggml_set_input(inp->tokens);
    inp->embd = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd_inp(), n_tokens);
    ggml_set_input(inp->embd);
    inp->h = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, hparams.n_embd, n_tokens);
    ggml_set_input(inp->h);
    ggml_set_name(inp->h, "mtp_h_input");

    ggml_tensor * tok_embd = ubatch.token ? ggml_get_rows(ctx0, model.tok_embd, inp->tokens) : inp->embd;
    cb(tok_embd, "mtp_tok_embd", il);
    ggml_tensor * h_norm = build_norm(inp->h, layer.nextn.hnorm, nullptr, LLM_NORM_RMS, il);
    ggml_tensor * e_norm = build_norm(tok_embd, layer.nextn.enorm, nullptr, LLM_NORM_RMS, il);
    res->add_input(std::move(inp));

    ggml_tensor * x = build_lora_mm(layer.nextn.eh_proj, ggml_concat(ctx0, e_norm, h_norm, 0));
    cb(x, "mtp_eh_proj", il);

    ggml_tensor * out_ids = build_inp_out_ids();
    auto * inp_attn = build_attn_inp_k();

    ggml_tensor * stream = ggml_repeat_4d(ctx0, ggml_reshape_3d(ctx0, x, n_embd, 1, n_tokens), n_embd, hc, n_tokens, 1);
    ggml_tensor * post;
    ggml_tensor * comb;
    ggml_tensor * cur = build_hc_pre(stream, layer.hc_attn_fn, layer.hc_attn_base, layer.hc_attn_chart, layer.attn_norm, &post, &comb, il);
    cur = build_mla_layer(cur, layer, inp_attn, il);
    stream = build_hc_post(stream, cur, post, comb, il);
    cur = build_hc_pre(stream, layer.hc_ffn_fn, layer.hc_ffn_base, layer.hc_ffn_chart, layer.ffn_norm, &post, &comb, il);
    // the MTP SwiGLU output exceeds the F16 range (1e5 seen), which an F16 ffn_down turns into inf on backends
    // that round the activation to half (Metal); a power-of-two scale around ffn_down is exact
    const float ffn_scale = 64.0f;
    cur = ggml_swiglu(ctx0, build_lora_mm(layer.ffn_up, cur));
    cb(cur, "ffn_swiglu", il);
    cur = build_lora_mm(layer.ffn_down, ggml_scale(ctx0, cur, 1.0f/ffn_scale));
    ggml_prec_set_acc(cur, GGML_PREC_F32);
    cur = ggml_scale(ctx0, cur, ffn_scale);
    cb(cur, "ffn_out", il);
    stream = build_hc_post(stream, cur, post, comb, il);

    cur = nullptr;
    for (int64_t i = 0; i < hc; ++i) {
        ggml_tensor * xi = ggml_view_2d(ctx0, stream, n_embd, n_tokens, stream->nb[2], i*stream->nb[1]);
        cur = cur ? ggml_add(ctx0, cur, xi) : xi;
    }
    cur = ggml_scale(ctx0, cur, 1.0f/hc);
    cur = build_norm(cur, layer.nextn.shared_head_norm, nullptr, LLM_NORM_RMS, il);
    cb(cur, "h_nextn", -1);

    ggml_tensor * cur_out = out_ids ? ggml_get_rows(ctx0, cur, out_ids) : cur;
    res->t_h_nextn = cparams.embeddings_nextn_masked ? cur_out : cur;
    ggml_build_forward_expand(gf, res->t_h_nextn);
    cb(cur_out, "result_norm", -1);
    res->t_embd = cur_out;

    cur = build_lora_mm(model.output, cur_out);
    cb(cur, "result_output", -1);
    res->t_logits = cur;
}
