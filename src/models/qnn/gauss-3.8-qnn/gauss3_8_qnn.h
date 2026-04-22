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

#include "quick_dot_ai_qnn_old.h"

namespace causallm {

/**
 * @brief Gauss3_8_QNN class
 * @note  This is the main class you register with the Factory.
 *
 */
class Gauss3_8_QNN : public Quick_Dot_AI_QNN_OLD {

public:
  static constexpr const char *architectures = "Gauss_3_8_QNN";

  Gauss3_8_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN_OLD(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Gauss3_8_QNN() = default;

  void initialize_input_outputs();

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  
  void run_with_embeddings(const void *prefill_embeds,
                                                 size_t n_tokens,
                                                 std::vector<int> seed_tokens,
                                                 bool do_sample,
                                                 bool log_output);

  const void *lookupEmbedding(int token_id) const;                                                 

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

  std::vector<ml::train::TensorDim::IO_TensorType> prefill_inputs;
  std::vector<ml::train::TensorDim::IO_TensorType> generation_inputs;

  uint16_t *input_sample_u16 = nullptr;
  uint16_t *generation_sample_u16 = nullptr;

  void *embedding_mmap_ptr = nullptr;
  size_t embedding_mmap_size = 0;
  size_t embedding_bytes_per_token = 0; // hidden_size * sizeof(uint16_t)

  // KV cache variables
  std::vector<uint16_t *> kvs;
  std::vector<int> kv_sizes;
  std::vector<uint16_t *> fresh_kvs;
};

} // namespace causallm

#endif /* __GAUSS_3_8_QNN_H__ */