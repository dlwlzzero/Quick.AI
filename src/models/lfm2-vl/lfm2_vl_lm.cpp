// SPDX-License-Identifier: Apache-2.0
#include "lfm2_vl_lm.h"

#include <factory.h>
#include <model_descriptor.h>

#include <memory>

namespace causallm {

Lfm2VlLM::Lfm2VlLM(json &cfg, json &generation_cfg, json &nntr_cfg)
  // Transformer is a VIRTUAL base (CausalLM : virtual public Transformer), so
  // the most-derived class must initialize it explicitly. Without this, when
  // Lfm2VlLM is the most-derived type the virtual base is default-constructed
  // (empty Transformer()), setupParameters() never runs, and BATCH_SIZE et al.
  // are left uninitialized — causing a runaway output_list allocation (OOM).
  // Mirror Lfm2CausalLM's own virtual-base init.
  : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    Lfm2CausalLM(cfg, generation_cfg, nntr_cfg) {
  emb_dim_ = cfg.value("hidden_size", 1024u);
  image_token_id_ = cfg.value("image_token_id", 396);
}

const void *Lfm2VlLM::lookupEmbedding(int token_id) const {
  if (token_id < 0)
    return nullptr;
  auto *self = const_cast<Lfm2VlLM *>(this);
  emb_scratch_ =
    self->Lfm2CausalLM::lookupEmbedding(static_cast<unsigned int>(token_id));
  if (emb_scratch_.size() != emb_dim_)
    return nullptr;
  return emb_scratch_.data();
}

__attribute__((constructor)) static void register_lfm2_vl_lm() {
  Factory::Instance().registerModel(
    "Lfm2ForCausalLM_VL", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<Lfm2VlLM>(cfg, generation_cfg, nntr_cfg);
    });

  static const ModelDescriptor d = {"lfm2-vl",
                                     "lfm2",
                                     "LFM2-VL 450M",
                                     QDA_RUNTIME_NATIVE,
                                     (1u << 0), // CPU
                                     QDA_CAP_MULTIMODAL | QDA_CAP_MESSAGES_API |
                                       QDA_CAP_STREAMING,
                                     "lfm2-vl",
                                     "Lfm2ForCausalLM_VL"};
  quick_dot_ai::register_model_descriptor(&d);
}

} // namespace causallm
