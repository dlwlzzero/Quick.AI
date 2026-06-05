// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <functional>
#include <string>
#include <unordered_map>

#include "quick_dot_ai_api.h"  // ErrorCode, CausalLmTokenCallback, CausalLmHandle

namespace causallm {
class Transformer;
}

/**
 * Per-architecture callbacks registered by gauss (or other proprietary) TU files.
 * When a gauss TU is absent (public build), no callbacks are registered for that
 * architecture; callers should fall back to CAUSAL_LM_ERROR_UNSUPPORTED.
 */
struct ModelCallbacks {
  /**
   * Apply architecture-specific chat template to a raw single-turn input.
   * Returns empty string if not registered (caller uses raw input).
   */
  std::function<std::string(const std::string &raw_input)> format_prompt;

  /** True when this architecture requires an HTP/QNN backend. */
  bool requires_htp = false;

  /**
   * Read the current KV-cache length from a loaded transformer.
   * Used for incremental-session tracking.
   * Returns 0 if not registered.
   */
  std::function<int(causallm::Transformer *model)> read_kv_len;

  /**
   * Given the full prompt history (already-formatted), extract the latest user
   * content and rebuild it as the minimal incremental prompt for next turn.
   * Returns empty string if not registered.
   */
  std::function<std::string(const std::string &full_prompt)> incremental_prompt;

  /**
   * Streaming multimodal execution.
   * `handle` is CausalLmHandle (= CausalLmModel*).
   * The gauss TU casts it to CausalLmModel* and accesses h.models[0]/[1].
   */
  std::function<ErrorCode(CausalLmHandle handle,
                          const float *pixel_values,
                          int num_patches, int orig_h, int orig_w,
                          const std::string &prompt,
                          CausalLmTokenCallback cb, void *user_data)>
    multimodal_streaming;

  /**
   * Blocking multimodal execution; appends generated text to *output.
   * `handle` is CausalLmHandle (= CausalLmModel*).
   */
  std::function<ErrorCode(CausalLmHandle handle,
                          const float *pixel_values,
                          int num_patches, int orig_h, int orig_w,
                          const std::string &prompt,
                          std::string *output)>
    multimodal_blocking;
};

/**
 * Registry keyed by architecture name string (e.g. "Gauss_3_8_QNN").
 * Gauss TUs call register_for() at static-init time.
 * quick_dot_ai_api.cpp calls lookup() at runtime.
 */
class ModelCallbackRegistry {
public:
  static ModelCallbackRegistry &instance();

  /** Register callbacks for one architecture name. */
  void register_for(const std::string &architecture, ModelCallbacks cb);

  /**
   * Look up callbacks for the given architecture.
   * Returns nullptr if not registered (non-gauss or gauss absent).
   */
  const ModelCallbacks *lookup(const std::string &architecture) const;

  /** True if ANY registered architecture has requires_htp = true. */
  bool any_requires_htp() const;

private:
  ModelCallbackRegistry() = default;
  ModelCallbackRegistry(const ModelCallbackRegistry &) = delete;
  ModelCallbackRegistry &operator=(const ModelCallbackRegistry &) = delete;

  std::unordered_map<std::string, ModelCallbacks> by_arch_;
};
