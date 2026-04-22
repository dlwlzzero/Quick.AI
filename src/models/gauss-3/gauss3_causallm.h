// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_causallm.h
 * @brief  Gauss3 CausalLM model extension using CausalLM standard layers
 * @note   This file demonstrates Gauss3 model for Quick.AI
 *         by extending the base CausalLM class from nntrainer.
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
 */

#ifndef __GAUSS_3_CAUSALLM_H__
#define __GAUSS_3_CAUSALLM_H__

#include <causal_lm.h>
#include <iostream>

namespace causallm {

/**
 * @brief Gauss3Transformer class
 * @note  Custom Transformer with Gauss3-specific sliding window pattern
 *        using CausalLM standard layers
 */
class Gauss3Transformer : virtual public Transformer {
public:
  static constexpr const char *architectures = "Gauss3Transformer";

  Gauss3Transformer(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Transformer(cfg, generation_cfg, nntr_cfg) {
    // Setup Gauss3-specific parameters
    try {
      NUM_SEQUENTIAL_LAYERS =
          cfg.contains("num_sequential_layers")
              ? cfg["num_sequential_layers"].get<unsigned int>()
              : 25;
      SLIDING_ATTENTION_KV_LAYER =
          cfg.contains("sliding_attention_kv_layer")
              ? cfg["sliding_attention_kv_layer"].get<unsigned int>() - 1
              : 22;
      FULL_ATTENTION_KV_LAYER =
          cfg.contains("full_attention_kv_layer")
              ? cfg["full_attention_kv_layer"].get<unsigned int>() - 1
              : 23;
      SLIDING_WINDOW_PATTERN =
          cfg.contains("sliding_window_pattern")
              ? cfg["sliding_window_pattern"].get<unsigned int>()
              : 5;
    } catch (const std::exception &e) {
      std::cerr << "Warning: Error setting up Gauss3 parameters: " << e.what()
                << std::endl;
      // Set defaults
      NUM_SEQUENTIAL_LAYERS = 25;
      SLIDING_ATTENTION_KV_LAYER = 22;
      FULL_ATTENTION_KV_LAYER = 23;
      SLIDING_WINDOW_PATTERN = 5;
    }
  }

  virtual ~Gauss3Transformer() = default;

  /**
   * @brief Construct model with Gauss3's sliding window pattern
   */
  void constructModel() override;

  /**
   * @brief Create custom transformer decoder block with sliding window pattern
   */
  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id, std::string input_name,
                                bool is_sliding,
                                std::string shared_states_name);

  /**
   * @brief Create custom attention using mha_core
   */
  std::vector<LayerHandle>
  createAttention(const int layer_id, int seq_len, int n_heads, int head_dim,
                  std::string query_name, std::string key_name,
                  std::string value_name, bool skip_prefill);

  /**
   * @brief Create custom MLP using fully_connected and swiglu
   */
  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim, std::string input_name,
                                     bool skip_prefill);

  /**
   * @brief Register custom layers (using CausalLM standard layers)
   */
  void registerCustomLayers() override;

protected:
  // Gauss3-specific parameters
  unsigned int NUM_SEQUENTIAL_LAYERS = 25;
  unsigned int SLIDING_ATTENTION_KV_LAYER = 22;
  unsigned int FULL_ATTENTION_KV_LAYER = 23;
  unsigned int SLIDING_WINDOW_PATTERN = 5;
};

/**
 * @brief Gauss3CausalLM class
 * @note  Main model class combining CausalLM and Gauss3Transformer
 */
class Gauss3CausalLM : public CausalLM, public Gauss3Transformer {

public:
  static constexpr const char *architectures = "Gauss3ForCausalLM";

  Gauss3CausalLM(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
        CausalLM(cfg, generation_cfg, nntr_cfg),
        Gauss3Transformer(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Gauss3CausalLM() = default;

  /**
   * @brief Construct model with Gauss3's sliding window pattern
   */
  void constructModel() override;

  /**
   * @brief Register custom layers
   */
  void registerCustomLayers() override;
};

} // namespace causallm

#endif /* __GAUSS_3_CAUSALLM_H__ */
