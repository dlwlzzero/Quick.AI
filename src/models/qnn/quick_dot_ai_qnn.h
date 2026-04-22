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

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif


#include "graph_parser.h"
#include <transformer.h>

#include <atomic>

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
    LOGD("--------------------------------- Quick_Dot_AI_QNN");
    setupParameters(cfg, generation_cfg, nntr_cfg);
  }

  ~Quick_Dot_AI_QNN() override;

  void initialize() override;

  void load_weight(const std::string &weight_path) override;

  void save_weight(const std::string &weight_path) override;

  void setupParameters(json &cfg, json &generation_cfg,
                       json &nntr_cfg) override;

  void constructModel() override;

  /**
   * @brief Attach (or detach) a BaseStreamer to intercept per-token output.
   *        Passing nullptr detaches any currently-attached streamer.
   */
  void setStreamer(::BaseStreamer *streamer) override { streamer_ = streamer; }

  /**
   * @brief Request cancellation of the current run().
   *
   * Thread-safe: sets the stop flag atomically, causing the token
   * generation loop to exit at the next token boundary. Safe to call
   * from any thread (e.g., from a UI cancel button handler).
   */
  void requestStop() override {
#ifdef __ANDROID__
    __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG,
                        "requestStop: setting stop_requested_ to true");
#else
    std::cout << "[DEBUG] requestStop: setting stop_requested_ to true" << std::endl;
#endif
    stop_requested_.store(true, std::memory_order_release);
  }

  /**
   * @brief Check if stop has been requested.
   * Thread-safe: can be called from any thread.
   */
  bool isStopRequested() const { return stop_requested_.load(std::memory_order_acquire); }

  /**
   * @brief Clear the stop request flag.
   * Thread-safe: can be called from any thread.
   */
  void clearStopRequest() { stop_requested_.store(false, std::memory_order_release); }

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
  std::string embedding_path;
  std::string binary_config_path;
  std::vector<std::string> graphs_to_use;

  // config
  int vocab_size;

  // Model map, key: graph name, value: QNN model info
  std::map<std::string, QNNModelInfo> models;

  bool uses_embedding = true;

  // Streaming support
  ::BaseStreamer *streamer_ = nullptr;
  std::string last_output_;

  /**
   * @brief Cooperative cancellation flag set by the attached streamer's
   *        put() returning non-zero, or by requestStop() from any thread.
   *        The token generation loop in run() checks this once per iteration
   *        and breaks out at the next safe boundary.
   *
   * Uses std::atomic for thread-safe access from any thread (e.g.,
   * cancel button handler in UI thread).
   */
  std::atomic<bool> stop_requested_{false};
};

} // namespace causallm

#endif /* __QUICK_DOT_AI_QNN_BASE_H__ */
