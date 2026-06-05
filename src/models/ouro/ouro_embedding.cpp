// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Seunghui Lee <shsh1004.lee@samsung.com>
 *
 * @file   ouro_embedding.cpp
 * @date   1 June 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seunghui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file defines OuroEmbedding's basic actions
 */

#include <algorithm>
#include <chrono>

#include <app_context.h>
#include <common.h>
#include <layer_context.h>
#include <mha_core.h>
#include <nntrainer_error.h>
#include <tensor.h>

#include <llm_util.hpp>
#include <ouro_embedding.h>
#include <factory.h>
#include <model_descriptor.h>

namespace causallm {

std::pair<Tensor, Tensor> OuroEmbedding::constructTransformerModule() {

  Tensor x =
    Tensor({1, 1, 1, static_cast<unsigned int>(INIT_SEQ_LEN)}, "input0");

  // Ouro: embed_tokens (vocab -> intermediate_size) + embed_projection
  // (intermediate_size -> hidden_size).
  const unsigned int EMBED_PROJ_DIM = INTERMEDIATE_SIZE;

  const std::string embedding_type =
    TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  LayerHandle embedding(createLayer(
    embedding_type,
    {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
     "weight_dtype=" + EMBEDDING_DTYPE,
     "out_dim=" + std::to_string(EMBED_PROJ_DIM),
     "scale=" + std::to_string(EMBEDDING_SCALE)}));
  Tensor h = embedding(x);

  LayerHandle embed_proj(createLayer(
    "fully_connected",
    {withKey("name", "embed_projection"), withKey("unit", DIM),
     withKey("disable_bias", "true"),
     withKey("weight_dtype", FC_LAYER_DTYPE)}));
  h = embed_proj(h);

  // current_ut_ steers the leaf-name / shared_from logic in OuroTransformer
  // overrides. Step k>0 leaves get layer<i>_ut<k>_<role> + shared_from=
  // layer<i>_<role>, so only one set of weights is stored in the pool.
  for (int ut = 0; ut < static_cast<int>(TOTAL_UT_STEPS); ++ut) {
    current_ut_ = ut;
    for (int i = 0; i < NUM_LAYERS; ++i) {
      h = createTransformerDecoderBlock(i, h);
    }
    h = applyOuroOutputNorm(ut, h);
  }
  current_ut_ = 0;

  // Pooling + normalize are appended by SentenceTransformer::constructModel
  // based on modules.json (1_Pooling, 2_Normalize).
  return {x, h};
}

std::vector<float *> OuroEmbedding::encode(const WSTR prompt,
                                           const WSTR system_prompt,
                                           const WSTR tail_prompt) {
  if (!is_initialized) {
    throw std::runtime_error("OuroEmbedding model is not initialized. Please "
                             "call initialize() before encode().");
  }

  std::string prompt_ = system_prompt + prompt + tail_prompt;
  auto _input = tokenizer->Encode(prompt_, true);

  unsigned int input_len =
    std::min(static_cast<unsigned int>(_input.size()),
             static_cast<unsigned int>(MAX_SEQ_LEN));

  std::vector<float> input_sample(static_cast<size_t>(BATCH_SIZE) * MAX_SEQ_LEN,
                                  0.0f);
  for (unsigned int b = 0; b < BATCH_SIZE; ++b) {
    for (unsigned int i = 0; i < input_len; ++i) {
      input_sample[static_cast<size_t>(b) * MAX_SEQ_LEN + i] =
        static_cast<float>(_input[i]);
    }
  }

  // 3-input mha_core mode: the graph has no cache_k_l<i> / cache_v_l<i>
  // placeholder inputs, so the only runtime input is the token-id buffer.
  // This bypasses SentenceTransformer::allocateAndBindKVCache which would
  // try to find non-existent placeholders.
  std::vector<float *> input;
  input.push_back(input_sample.data());

  std::vector<float *> label;

  // Time the single prefill forward pass and populate prefill telemetry, the
  // same way SentenceTransformer::encode does. OuroEmbedding overrides encode()
  // to skip external-KV-cache binding, so without this the "Embedding with
  // NNTrainer" report shows "prefill: 0 tokens, 0 ms, 0 TPS". For an embedding
  // model this is one forward pass (no autoregressive generation), so the
  // reported TPS is prefill throughput (input_len / prefill_ms), not token gen.
  auto start_prefill = std::chrono::high_resolution_clock::now();
  std::vector<float *> output = model->incremental_inference(
    BATCH_SIZE, input, label, input_len, 0, input_len, false);
  auto finish_prefill = std::chrono::high_resolution_clock::now();

  auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    finish_prefill - start_prefill);
  performance_metrics.prefill_tokens = input_len;
  performance_metrics.prefill_duration_ms = prefill_duration.count();

  return output;
}

} // namespace causallm

/**
 * @brief Auto-registration via constructor attribute
 *
 */
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "OuroModel", [](causallm::json cfg, causallm::json generation_cfg,
                              causallm::json nntr_cfg) {
        return std::make_unique<causallm::OuroEmbedding>(cfg, generation_cfg,
                                                          nntr_cfg);
      });

  // Self-register the catalog entry (same pattern as gauss-3.8-qnn). This is
  // what makes Ouro show up in get_model_catalog_json() so the Android app can
  // list and select it. Ouro is a CPU nntrainer sentence-embedding model
  // (1_Pooling + 2_Normalize, model_type=embedding), so: backend CPU (bit 0),
  // capability EMBEDDING, arch_string "OuroModel". The model is loaded from an
  // on-device dir (external config.json / nntr_config.json), so config_name is
  // just the catalog id.
  static const ModelDescriptor d = {
    "ouro",       "ouro",
    "Ouro",       QDA_RUNTIME_NATIVE,
    (1u << 0),    QDA_CAP_EMBEDDING,
    "ouro",       "OuroModel"};
  quick_dot_ai::register_model_descriptor(&d);
}

