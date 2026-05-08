// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3.8_qnn.h
 * @brief  QNN model extension template
 * @note   This file demonstrates how to create a custom QNN Quick.AI model
 *         by extending the base CausalLM class from nntrainer.
 *
 */

#ifndef __GAUSS_3_8_QNN_H__
#define __GAUSS_3_8_QNN_H__

#include "quick_dot_ai_qnn.h"

#include <cstddef>
#include <cstdint>
#include <utility>

namespace causallm {

/**
 * @brief Gauss3_8_QNN class
 * @note  This is the main class you register with the Factory.
 *
 */
class Gauss3_8_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "Gauss_3_8_QNN";

  Gauss3_8_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {
    LOGD("Gauss 3.8 parameters set up ");
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Gauss3_8_QNN() override;

  void initialize() override;

  void initialize_kv_cache();

  void setupParameters(json &cfg, json &generation_cfg, json &nntr_cfg) override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  void run_with_embeddings(const void *prefill_embeds, size_t n_tokens,
                           std::vector<int> seed_tokens, bool do_sample,
                           bool log_output);

  const void *lookupEmbedding(int token_id) const;

  size_t embeddingBytesPerToken() const { return embedding_bytes_per_token; }
  std::pair<float, int> get_embedding_info();

private:
  struct KvOutputBinding {
    int output_index;
    int kv_index;
    int layer_index;
    bool is_key;
  };

  void reset_prefill_kv_cache_inputs();
  void sync_generation_kv_cache_to_prefill();
  std::string normalize_conversation_prompt(const std::string &prompt) const;
  void append_outputs_to_kv_cache(
      const std::vector<ml::train::TensorDim::IO_TensorType> &step_outputs,
      const std::vector<KvOutputBinding> &bindings, int target_position,
      int rows, int src_row_length, const std::string &graph_name);

  // Input/output tensors
  uint16_t *attention_mask = nullptr;
  uint16_t *sliding_attention_mask = nullptr;
  uint16_t *generation_attention_mask = nullptr;
  uint16_t *generation_sliding_attention_mask = nullptr;

  uint16_t *position_ids_cos = nullptr;
  uint16_t *position_ids_sin = nullptr;
  uint16_t *swa_position_ids_cos = nullptr;
  uint16_t *swa_position_ids_sin = nullptr;
  uint16_t *prefill_position_ids_cos = nullptr;
  uint16_t *prefill_position_ids_sin = nullptr;
  uint16_t *prefill_swa_position_ids_cos = nullptr;
  uint16_t *prefill_swa_position_ids_sin = nullptr;
  uint16_t *generation_position_ids_cos = nullptr;
  uint16_t *generation_position_ids_sin = nullptr;
  uint16_t *generation_swa_position_ids_cos = nullptr;
  uint16_t *generation_swa_position_ids_sin = nullptr;

  float *input_sample = nullptr;
  float *generation_sample = nullptr;

  uint16_t *input_sample_u16 = nullptr;
  uint16_t *generation_sample_u16 = nullptr;

  // KV cache variables
  int kv_len = 0;
  bool conversation_started_ = false;

  int generation_logits_output_index = -1;
  int prefill_attention_mask_elements = 0;
  int prefill_sliding_attention_mask_elements = 0;
  int generation_attention_mask_elements = 0;
  int generation_sliding_attention_mask_elements = 0;
  int generation_full_kv_past_length = 0;
  int generation_sliding_kv_past_length = 0;
  int rope_cache_seq_len = 0;

  std::vector<uint8_t *> kvs;
  std::vector<int> kv_sizes;
  std::vector<uint8_t *> fresh_kvs;
  std::vector<int> kv_row_lengths;

  std::vector<uint8_t *> prefill_kvs;
  std::vector<int> prefill_kv_sizes;
  std::vector<int> prefill_kv_row_lengths;
  std::vector<int> prefill_to_generation_kv_indices;
  std::vector<int> prefill_kv_is_key;

  std::vector<KvOutputBinding> prefill_output_kv_bindings;
  std::vector<KvOutputBinding> generation_output_kv_bindings;

  // Language model specific variables
  int num_hidden_layers;
  int max_window_layers;
  int hidden_size;
  int sequence_length;
  int max_seq_len;
  int sliding_window;
  float local_rope_theta;
  float rope_theta;
  int context_size;
  int pos_dim;
  int head_dim;

  // generation_config
  int padding_token;
  int eos_token;
  int top_k;
  float top_p;
  float temperature;
  float repetition_penalty;
  float logit_scale;
  int logit_offset;

  // LoRA path (optional)
  std::string lora_path;

  // mmap-backed pre-quantized text embedding table for uses_embedding=false.
  void *embedding_mmap_ptr = nullptr;
  size_t embedding_mmap_size = 0;
  size_t embedding_bytes_per_token = 0;
};

} // namespace causallm

#endif /* __GAUSS_3_8_QNN_H__ */
