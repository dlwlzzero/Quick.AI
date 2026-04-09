// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3.6_qnn.h
 * @brief  QNN model extension template
 * @note   This file demonstrates how to create a custom QNN Quick.AI model
 *         by extending the base CausalLM class from nntrainer.
 *
 */

#ifndef __GAUSS_3_6_QNN_H__
#define __GAUSS_3_6_QNN_H__

#include "quick_dot_ai_qnn.h"

namespace causallm {

/**
 * @brief Gauss3_6_QNN class
 * @note  This is the main class you register with the Factory.
 *
 */
class Gauss3_6_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "Gauss_3_6_QNN";

  Gauss3_6_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {}

  virtual ~Gauss3_6_QNN() = default;
};

} // namespace causallm

#endif /* __GAUSS_3_6_QNN_H__ */