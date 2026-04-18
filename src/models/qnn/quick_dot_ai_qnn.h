// SPDX-License-Identifier: Apache-2.0
/**
 * @file   quick_dot_ai_qnn_base.h
 * @brief  QNN model for Quick.AI template
 * @note   This file implements a layer that executes QNN binary file within
 * transformer.h architecture.
 *
 */

#ifndef __QUICK_DOT_AI_QNN_H__
#define __QUICK_DOT_AI_QNN_H__

#include "graph_parser.h"
#include <transformer.h>

namespace causallm {
/**
 * @brief QNN Model info
 * @note  This struct contains QNN model information for execution.
 */
struct QNNModelInfo {
  GraphInfo graph_info;
  ModelHandle model_handle;
  std::vector<ml::train::TensorDim::IO_TensorType> model_inputs;
};

/**
 * @brief Quick_Dot_AI_QNN_Base class
 * @note  This is the base class for QNN.
 */
class Quick_Dot_AI_QNN : public Transformer {

public:
  Quick_Dot_AI_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::MODEL) {
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Quick_Dot_AI_QNN() override;

  void initialize() override;

  void load_weight(const std::string &weight_path) override;

  void save_weight(const std::string &weight_path) override;

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
  std::string model_file_name;
  std::string binary_config_path;
  std::vector<std::string> graphs_to_use;

  // config
  int vocab_size;

  // Model map, key: graph name, value: QNN model info
  std::map<std::string, QNNModelInfo> models;

  bool uses_embedding = true;
};

} // namespace causallm

#endif /* __QUICK_DOT_AI_QNN_BASE_H__ */
