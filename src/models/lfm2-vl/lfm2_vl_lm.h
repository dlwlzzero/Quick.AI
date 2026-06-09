// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_lm.h
 * @brief  LFM2 LM consumer for the LFM2-VL composite. Adds the base-Transformer
 *         composer virtuals on top of causallm::Lfm2CausalLM.
 */
#ifndef __LFM2_VL_LM_H__
#define __LFM2_VL_LM_H__

#include <lfm2_causallm.h>

#include <utility>
#include <vector>

namespace causallm {

class Lfm2VlLM : public Lfm2CausalLM {
public:
  Lfm2VlLM(json &cfg, json &generation_cfg, json &nntr_cfg);

  size_t embeddingBytesPerToken() const override {
    return static_cast<size_t>(emb_dim_) * sizeof(float);
  }

  const void *lookupEmbedding(int token_id) const override;

  std::pair<float, int> get_embedding_info() override { return {1.0f, 0}; }

  int imagePlaceholderTokenId() const override { return image_token_id_; }

private:
  unsigned int emb_dim_ = 1024;   // text_config hidden_size
  int image_token_id_ = 396;      // config image_token_id
  mutable std::vector<float> emb_scratch_;
};

} // namespace causallm

#endif // __LFM2_VL_LM_H__
