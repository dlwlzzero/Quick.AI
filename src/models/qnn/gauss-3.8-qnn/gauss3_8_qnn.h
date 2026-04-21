// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_8_qnn.h
 * @brief  QNN model extension template
 * @note   This file demonstrates how to create a custom QNN Quick.AI model
 *         by extending the Quick_Dot_AI_QNN_Base class.
 *
 */

#ifndef __GAUSS_3_8_QNN_H__
#define __GAUSS_3_8_QNN_H__

#include "quick_dot_ai_qnn.h"


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
    LOGD("Gauss 3.8 asdfasdfasdfasdfasdfasdfsdfasdf ");
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  virtual ~Gauss3_8_QNN();

  void initialize();

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  /**
   * @brief Multimodal inference entry: consume pre-computed uint16
   *        embeddings as prefill input instead of tokenizing a prompt.
   *
   * Requires uses_embedding=false so that "inputs_embeds" is a regular
   * uint16 tensor. Caller composes the sequence (e.g.
   * [text_pre | image_embeds | text_post]) before calling.
   *
   * @param prefill_embeds  uint16 buffer of [n_tokens × hidden_size].
   * @param n_tokens        Number of tokens in prefill_embeds.
   * @param seed_tokens     Token IDs for the repetition-penalty window.
   *                        Image slots may use any placeholder id.
   * @param do_sample       If false, argmax decoding.
   * @param log_output      Echo to stdout when no streamer is attached.
   */
  void run_with_embeddings(const void *prefill_embeds, size_t n_tokens,
                           std::vector<int> seed_tokens,
                           bool do_sample = false, bool log_output = true);

  /**
   * @brief Look up a single token's pre-quantized embedding vector.
   * @return Pointer into the mmap'd embedding table, or nullptr if the
   *         table is not loaded or token_id is out of range. Buffer
   *         size = embeddingBytesPerToken() bytes.
   */
  const void *lookupEmbedding(int token_id) const;

  /// @brief bytes-per-token for external embedding composition.
  size_t embeddingBytesPerToken() const { return embedding_bytes_per_token; }

private:
  // Input/output tensors
  uint16_t *attention_mask;
  uint16_t *sliding_attention_mask;
  uint16_t *generation_attention_mask;
  uint16_t *generation_sliding_attention_mask;

  uint16_t *position_ids_cos;
  uint16_t *position_ids_sin;
  uint16_t *swa_position_ids_cos;
  uint16_t *swa_position_ids_sin;
  uint16_t *prefill_position_ids_cos;
  uint16_t *prefill_position_ids_sin;
  uint16_t *prefill_swa_position_ids_cos;
  uint16_t *prefill_swa_position_ids_sin;
  uint16_t *generation_position_ids_cos;
  uint16_t *generation_position_ids_sin;
  uint16_t *generation_swa_position_ids_cos;
  uint16_t *generation_swa_position_ids_sin;

  float *input_sample;
  float *generation_sample;

  // uses_embedding=false path: inputs_embeds is uint16 instead of
  // token-id float. Keep both pointer types; only one is active
  // depending on configuration.
  uint16_t *input_sample_u16 = nullptr;
  uint16_t *generation_sample_u16 = nullptr;

  // mmap-backed pre-quantized text embedding table. Loaded lazily in
  // initialize() when uses_embedding=false. Used both by the external
  // multimodal composer (via lookupEmbedding) and by this class's
  // generation loop to fetch the next token's embedding per step.
  void *embedding_mmap_ptr = nullptr;
  size_t embedding_mmap_size = 0;
  size_t embedding_bytes_per_token = 0;  // hidden_size * sizeof(uint16_t)

  // KV cache variables
  std::vector<uint16_t *> kvs;
  std::vector<int> kv_sizes;
  std::vector<uint16_t *> fresh_kvs;

  // Language model specific variables
  // Config
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
};

} // namespace causallm

#endif /* __GAUSS_3_8_QNN_H__ */
