// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_6_qnn.cpp
 * @brief  QNN model implementation with self-registration
 * @note   This model auto-registers with the CausalLM Factory via
 * __attribute__((constructor)) when linked or loaded.
 *
 *         No modification to nntrainer's main.cpp is needed.
 */

#include "gauss3_6_qnn.h"
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

#include <iostream>

/**
 * @brief Auto-registration via constructor attribute
 *
 * This function runs automatically when the shared library is loaded
 * (before main()). It registers all custom models with the CausalLM Factory.
 *
 */
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "Gauss_3_6_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                          causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gauss3_6_QNN>(cfg, generation_cfg,
                                                        nntr_cfg);
      });

  // Add more custom models here:
  // causallm::Factory::Instance().registerModel(
  //   "MyNewModelForCausalLM",
  //   [](causallm::json cfg, causallm::json generation_cfg,
  //      causallm::json nntr_cfg) {
  //     return std::make_unique<causallm::MyNewModel>(cfg, generation_cfg,
  //                                                   nntr_cfg);
  //   });
}