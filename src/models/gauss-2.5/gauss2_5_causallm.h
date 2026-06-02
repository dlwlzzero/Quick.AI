// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss2.5_causallm.h
 * @brief  Custom CausalLM model extension template
 * @note   This file demonstrates how to create a custom CausalLM model
 *         by extending the base CausalLM class from nntrainer.
 *
 * To create your own model:
 *   1. Inherit from CausalLM (and optionally create a Transformer variant)
 *   2. Override createAttention() to customize the attention mechanism
 *   3. Override createMlp() to customize the feed-forward network
 *   4. Override registerCustomLayers() to register any custom layers
 *   5. Override setupParameters() to add model-specific parameters
 *   6. Add Factory::Instance().registerModel() in the constructor attribute
 *      at the bottom of the .cpp file (auto-registers when .so is loaded)
 */

#ifndef __GAUSS_2_5_CAUSALLM_H__
#define __GAUSS_2_5_CAUSALLM_H__

#include <causal_lm.h>

namespace causallm {

/**
 * @brief Gauss2_5_Causallm class
 * @note  This is the main class you register with the Factory.
 *        It combines CausalLM (for generation logic) with
 *        CustomTransformer (for the model architecture).
 *
 *        The diamond inheritance pattern is:
 *          Transformer
 *           /      \
 *      CausalLM  CustomTransformer
 *           \      /
 *        Gauss2_5_Causallm
 */
class Gauss2_5_Causallm : public CausalLM {

public:
  static constexpr const char *architectures = "GaussForCausalLM";

  Gauss2_5_Causallm(json &cfg, json &generation_cfg, json &nntr_cfg) :
    Transformer(cfg, generation_cfg, nntr_cfg, ModelType::CAUSALLM),
    CausalLM(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Gauss2_5_Causallm() = default;
};

} // namespace causallm

#endif /* __GAUSS_2_5_CAUSALLM_H__ */