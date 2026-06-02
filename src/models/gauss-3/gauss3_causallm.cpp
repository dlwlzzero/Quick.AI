// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_causallm.cpp
 * @brief  Gauss3 CausalLM model implementation with self-registration
 * @note   This model auto-registers with the CausalLM Factory via
 *         __attribute__((constructor)) when linked or loaded.
 *
 *         Gauss3 uses CausalLM standard layers:
 *         - mha_core: Multi-head attention with rope_theta, sliding_window
 *         - rms_norm: Standard RMS normalization
 *         - swiglu: SwiGLU activation
 *         - fully_connected: Standard FC layers
 *
 *         Gauss3-specific features:
 *         - Sliding window pattern with shared KV states
 *         - Dynamic theta parameter (500000 for sliding, 2000000 for full)
 *
 *         No modification to nntrainer's main.cpp is needed.
 */

#include <gauss3_causallm.h>
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

#include <iostream>

namespace causallm {

/**
 * @brief Register custom layers (using CausalLM standard layers)
 */
void Gauss3Transformer::registerCustomLayers() {
  // CausalLM standard layers are already registered in nntrainer
}

/**
 * @brief Construct model with Gauss3's sliding window pattern
 */
void Gauss3Transformer::constructModel() {
  // layers used in the model
  std::vector<LayerHandle> layers;

  // create model
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  // create input layer
  layers.push_back(createLayer(
      "input",
      {withKey("name", "input0"),
       withKey("input_shape", "1:1:" + std::to_string(INIT_SEQ_LEN))}));

  const std::string embedding_type =
      TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "embedding_layer";

  // create embedding layer (using tie_word_embedding)
  layers.push_back(createLayer(
      embedding_type,
      {"name=embedding0", "in_dim=" + std::to_string(NUM_VOCAB),
       "weight_dtype=" + EMBEDDING_DTYPE, "out_dim=" + std::to_string(DIM),
       "scale=" + std::to_string(EMBEDDING_SCALE)}));

  // create sequential decoder layers (0 to NUM_SEQUENTIAL_LAYERS-1)
  for (int i = 0; i < (int)NUM_SEQUENTIAL_LAYERS; ++i) {
    bool is_sliding = (i + 1) % SLIDING_WINDOW_PATTERN != 0;
    std::vector<LayerHandle> transformer;
    if (i == 0)
      transformer =
          createTransformerDecoderBlock(0, "embedding0", is_sliding, "");
    else
      transformer = createTransformerDecoderBlock(
          i, "layer" + std::to_string(i - 1) + "_decoder_output", is_sliding,
          "");
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // Create KV shared state layers (Gauss3 v3 feature)
  std::string full_shared_states_name = "full_shared_states";
  std::string sliding_shared_states_name = "sliding_shared_states";

  layers.push_back(createLayer(
      "rms_norm", {withKey("name", full_shared_states_name),
                   withKey("epsilon", std::to_string(NORM_EPS)),
                   withKey("input_layers",
                           "layer" + std::to_string(FULL_ATTENTION_KV_LAYER) +
                               "_decoder_output"),
                   withKey("packed", "false")}));

  layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", sliding_shared_states_name),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("input_layers", "layer" +
                                   std::to_string(SLIDING_ATTENTION_KV_LAYER) +
                                   "_decoder_output"),
       withKey("packed", "false")}));

  // Create shared decoder layers (NUM_SEQUENTIAL_LAYERS to NUM_LAYERS-1)
  for (int i = NUM_SEQUENTIAL_LAYERS; i < NUM_LAYERS; ++i) {
    bool is_sliding = ((i + 1) % SLIDING_WINDOW_PATTERN) != 0;
    std::vector<LayerHandle> transformer;
    if (is_sliding) {
      transformer = createTransformerDecoderBlock(
          i, "layer" + std::to_string(i - 1) + "_decoder_output", is_sliding,
          sliding_shared_states_name);
    } else {
      transformer = createTransformerDecoderBlock(
          i, "layer" + std::to_string(i - 1) + "_decoder_output", is_sliding,
          full_shared_states_name);
    }
    layers.insert(layers.end(), transformer.begin(), transformer.end());
  }

  // create rms_norm
  layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "output_norm"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("input_layers",
               "layer" + std::to_string(NUM_LAYERS - 1) + "_decoder_output"),
       withKey("skip_prefill", "true"), withKey("packed", "false")}));

  // add created layers into the model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }
}

/**
 * @brief Create custom transformer decoder block with sliding window pattern
 */
std::vector<LayerHandle> Gauss3Transformer::createTransformerDecoderBlock(
    const int layer_id, std::string input_name, bool is_sliding,
    std::string shared_states_name) {

  std::vector<LayerHandle> layers;
  std::vector<LayerHandle> att_layer;

  // Input Layer Norm (using rms_norm)
  layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_attention_norm"),
       withKey("input_layers", input_name),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("skip_prefill", shared_states_name.empty() ? "false" : "true"),
       withKey("packed", "false")}));

  // Query: always uses rms_norm output
  // Key, Value: uses rms_norm if no shared states, otherwise uses shared states
  std::string query_input =
      "layer" + std::to_string(layer_id) + "_attention_norm";
  std::string kv_input =
      shared_states_name.empty()
          ? "layer" + std::to_string(layer_id) + "_attention_norm"
          : shared_states_name;
  bool skip_prefill = !shared_states_name.empty();
  att_layer =
      createAttention(layer_id, INIT_SEQ_LEN, NUM_HEADS, DIM / NUM_HEADS,
                      query_input, kv_input, kv_input, skip_prefill);
  layers.insert(layers.end(), att_layer.begin(), att_layer.end());

  // Residual 1
  layers.push_back(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_add"),
       withKey("input_layers", input_name + ",layer" +
                                   std::to_string(layer_id) +
                                   "_attention_out")}));

  // Post Attention Layer Norm (using rms_norm)
  layers.push_back(createLayer(
      "rms_norm",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_norm"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_decoder_add"),
       withKey("epsilon", std::to_string(NORM_EPS)),
       withKey("skip_prefill", shared_states_name.empty() ? "false" : "true"),
       withKey("packed", "false")}));

  // FFN
  auto ffn_layer = createMlp(layer_id, DIM, INTERMEDIATE_SIZE,
                             "layer" + std::to_string(layer_id) + "_ffn_norm",
                             !shared_states_name.empty());
  layers.insert(layers.end(), ffn_layer.begin(), ffn_layer.end());

  // Residual 2
  layers.push_back(createLayer(
      "addition",
      {withKey("name", "layer" + std::to_string(layer_id) + "_decoder_output"),
       withKey("input_layers", "layer" + std::to_string(layer_id) +
                                   "_decoder_add,layer" +
                                   std::to_string(layer_id) + "_ffn_down")}));

  return layers;
}

/**
 * @brief Create custom attention using mha_core (CausalLM standard layer)
 */
std::vector<LayerHandle>
Gauss3Transformer::createAttention(const int layer_id, int seq_len, int n_heads,
                                   int head_dim, std::string query_name,
                                   std::string key_name, std::string value_name,
                                   bool skip_prefill) {

  std::vector<LayerHandle> layers;
  auto Q = "layer" + std::to_string(layer_id) + "_wq";
  auto K = "layer" + std::to_string(layer_id) + "_wk";
  auto V = "layer" + std::to_string(layer_id) + "_wv";
  auto A = "layer" + std::to_string(layer_id) + "_attention";
  auto O = "layer" + std::to_string(layer_id) + "_attention_out";

  // Determine theta parameter based on sliding/full attention
  // This is calculated from the pattern: sliding if (layer_id+1) % 5 != 0
  bool is_sliding = ((layer_id + 1) % SLIDING_WINDOW_PATTERN) != 0;
  float dynamic_theta = is_sliding ? 500000.f : 2000000.f;

  // Q layer (using fully_connected)
  std::vector<std::string> q_params = {
      withKey("name", Q),
      withKey("unit", head_dim * n_heads),
      withKey("disable_bias", "true"),
      withKey("input_layers", query_name),
      withKey("weight_initializer", "ones"),
      withKey("skip_prefill", skip_prefill ? "true" : "false")};
  layers.push_back(createLayer("fully_connected", q_params));

  // K layer (using fully_connected)
  std::vector<std::string> k_params = {
      withKey("name", K), withKey("unit", head_dim * n_heads / GQA_SIZE),
      withKey("disable_bias", "true"), withKey("input_layers", key_name),
      withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", k_params));

  // V layer (using fully_connected)
  std::vector<std::string> v_params = {
      withKey("name", V), withKey("unit", head_dim * n_heads / GQA_SIZE),
      withKey("disable_bias", "true"), withKey("input_layers", value_name),
      withKey("weight_initializer", "ones")};
  layers.push_back(createLayer("fully_connected", v_params));

  // Attention core layer (using mha_core - CausalLM standard layer)
  // Note: sliding_window and rope_theta are supported by mha_core
  std::vector<std::string> a_params;
  a_params.push_back(withKey("name", A));
  a_params.push_back(withKey("num_heads", n_heads));
  a_params.push_back(withKey("num_heads_kv", n_heads / GQA_SIZE));
  a_params.push_back(
      withKey("max_timestep", std::to_string(INIT_SEQ_LEN + NUM_TO_GENERATE)));
  a_params.push_back(
      withKey("sliding_window", is_sliding ? SLIDING_WINDOW : UINT_MAX));
  a_params.push_back(withKey("rope_theta", dynamic_theta));
  a_params.push_back(
      withKey("max_new_tokens", std::to_string(NUM_TO_GENERATE)));
  a_params.push_back(withKey("max_position_embeddings",
                             std::to_string(MAX_POSITION_EMBEDDINGS)));
  a_params.push_back(withKey("input_layers", Q + "," + K + "," + V));
  a_params.push_back(withKey("is_causal", "true"));
  a_params.push_back(withKey("skip_prefill", skip_prefill ? "true" : "false"));
  layers.push_back(createLayer("mha_core", a_params));

  // O layer (using fully_connected)
  std::vector<std::string> o_params = {
      withKey("name", O),
      withKey("unit", head_dim * n_heads),
      withKey("disable_bias", "true"),
      withKey("input_layers", A),
      withKey("weight_initializer", "ones"),
      withKey("skip_prefill", skip_prefill ? "true" : "false")};
  layers.push_back(createLayer("fully_connected", o_params));

  return layers;
}

/**
 * @brief Create custom MLP using fully_connected and swiglu (CausalLM standard
 * layers)
 */
std::vector<LayerHandle> Gauss3Transformer::createMlp(const int layer_id,
                                                      int dim, int hidden_dim,
                                                      std::string input_name,
                                                      bool skip_prefill) {
  std::vector<LayerHandle> layers;

  // Create FFN UP layer
  layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_up"),
       withKey("unit", hidden_dim), withKey("disable_bias", "true"),
       withKey("input_layers", input_name),
       withKey("weight_initializer", "ones"),
       withKey("skip_prefill", skip_prefill ? "true" : "false")}));

  // Create FFN GATE layer
  layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_gate"),
       withKey("unit", hidden_dim), withKey("disable_bias", "true"),
       withKey("input_layers", input_name),
       withKey("weight_initializer", "ones"),
       withKey("skip_prefill", skip_prefill ? "true" : "false")}));

  // Create SwiGLU activation (using swiglu - CausalLM standard layer)
  layers.push_back(createLayer(
      "swiglu",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
       withKey("input_layers", "layer" + std::to_string(layer_id) +
                               "_ffn_gate,layer" + std::to_string(layer_id) +
                               "_ffn_up"),
       withKey("skip_prefill", skip_prefill ? "true" : "false")}));

  // Create output projection (using fully_connected)
  layers.push_back(createLayer(
      "fully_connected",
      {withKey("name", "layer" + std::to_string(layer_id) + "_ffn_down"),
       withKey("unit", dim), withKey("disable_bias", "true"),
       withKey("input_layers",
               "layer" + std::to_string(layer_id) + "_ffn_swiglu"),
       withKey("weight_initializer", "ones"),
       withKey("skip_prefill", skip_prefill ? "true" : "false")}));

  return layers;
}

/**
 * @brief Construct model with Gauss3's sliding window pattern
 */
void Gauss3CausalLM::constructModel() {
  Gauss3Transformer::constructModel();
  const std::string lmhead_type =
      TIE_WORD_EMBEDDINGS ? "tie_word_embeddings" : "lm_head";

  // add lmhead
  std::vector<std::string> lmhead_prop = {
      withKey("name", "output_of_causallm"),
      withKey("unit", NUM_VOCAB),
      withKey("disable_bias", "true"),
      withKey("input_layers", "output_norm"),
      withKey("weight_dtype", LMHEAD_DTYPE),
      withKey("skip_prefill", "true"),
  };

  if (TIE_WORD_EMBEDDINGS)
    lmhead_prop.emplace_back(withKey("shared_from", "embedding0"));

  model->addLayer(createLayer(lmhead_type, lmhead_prop));
}

/**
 * @brief Register custom layers for Gauss3CausalLM
 */
void Gauss3CausalLM::registerCustomLayers() {
  CausalLM::registerCustomLayers();
  Gauss3Transformer::registerCustomLayers();
}

} // namespace causallm

/**
 * @brief Auto-registration via constructor attribute
 *
 */
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "Gauss3ForCausalLM", [](causallm::json cfg, causallm::json generation_cfg,
                              causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gauss3CausalLM>(cfg, generation_cfg,
                                                          nntr_cfg);
      });
}
