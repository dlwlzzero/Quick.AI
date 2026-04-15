// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn.h
 * @brief  QNN model for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 *
 */

#ifndef __QUICK_DOT_AI_QNN_H__
#define __QUICK_DOT_AI_QNN_H__

#include <transformer.h>

namespace causallm {

/**
 * @brief Gauss2_5_Causallm class
 * @note  This is the main class you register with the Factory.
 *        It combines CausalLM (for generation logic) with
 *        CustomTransformer (for the model architecture).
 *        Unlike CausalLM/CustomTransformer, this class does not us
 *        diamond interitance pattern.
 */
class Quick_Dot_AI_QNN : public Transformer {

public:
  Quick_Dot_AI_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Quick_Dot_AI_QNN() = default;

  void initialize() override;

  void initialize(const std::string &native_lib_dir) override;

  virtual void initialize_input_outputs() = 0;

  void load_weight(const std::string &weight_path) override;

  void save_weight(const std::string &weight_path) override;

  virtual void run(const WSTR prompt, bool do_sample = false,
                   const WSTR system_prompt = "", const WSTR tail_prompt = "",
                   bool log_output = true) = 0;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void constructModel() override;

  std::vector<LayerHandle>
  createTransformerDecoderBlock(const int layer_id,
                                std::string input_name) override;

  std::vector<LayerHandle> createAttention(const int layer_id, int seq_len,
                                           int n_heads, int head_dim,
                                           std::string query_name,
                                           std::string key_name,
                                           std::string value_name) override;

  std::vector<LayerHandle> createMlp(const int layer_id, int dim,
                                     int hidden_dim,
                                     std::string input_name) override;

  void registerCustomLayers() override;

protected:
  // nntr_config
  std::string model_path;
  std::string embedding_path;
  std::string tokenizer_path;

  // config
  // TODO struct?
  std::string prefill_graph_name;
  std::string prefill_input_names;
  std::string prefill_output_names;
  std::string prefill_in_quant;
  std::string prefill_out_quant;
  std::string prefill_in_dim;
  std::string prefill_out_dim;
  std::string prefill_in_data_format;
  std::string prefill_out_data_format;
  std::string prefill_out_tensor_format;
  std::vector<std::string> prefill_non_embed_input_names;
  std::vector<std::string> prefill_non_embed_input_dims;
  ModelHandle prefill_model;

  std::string generation_graph_name;
  std::string generation_input_names;
  std::string generation_output_names;
  std::string generation_in_quant;
  std::string generation_out_quant;
  std::string generation_in_dim;
  std::string generation_out_dim;
  std::string generation_in_data_format;
  std::string generation_out_data_format;
  std::string generation_out_tensor_format;
  std::vector<std::string> generation_non_embed_input_names;
  std::vector<std::string> generation_non_embed_input_dims;
  ModelHandle generation_model;

  int num_hidden_layers;
  int max_window_layers;
  int hidden_size;
  int sequence_length;
  int vocab_size;
  int max_seq_len;
  int sliding_window;
  float local_rope_theta;
  float rope_theta;
  int context_size;
  int pos_dim;
  int head_dim;
  std::vector<int> lora_sizes;

  // generation_config
  int padding_token;
  int eos_token;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float logit_scale;
  int logit_offset;
};

} // namespace causallm

#endif /* __QUICK_DOT_AI_QNN_H__ */