// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file    quick_dot_ai_api.h
 * @date    20 Mar 2026
 * @brief   C API for src (extension of CausalLM)
 *
 *          This header is self-contained: if causal_lm_api.h has already
 *          been included its types are reused; otherwise fallback
 *          definitions are provided so that this single header is
 *          sufficient for application code.
 *
 * @see     https://github.com/nntrainer/nntrainer
 * @author  Eunju Yang <ej.yang@samsung.com>
 * @bug     No known bugs except for NYI items
 */
#ifndef __QUICK_DOT_AI_API_H__
#define __QUICK_DOT_AI_API_H__

/* ── Extended model types (src additions) ────────────────────── */

#ifdef __CAUSAL_LM_API_H__

#ifndef CAUSAL_LM_MODEL_GAUSS2_5
#define CAUSAL_LM_MODEL_GAUSS2_5 ((ModelType)1)
#endif

#else /* causal_lm_api.h not included — provide full definitions */

#define __CAUSAL_LM_API_H__

#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>

typedef enum {
  CAUSAL_LM_ERROR_NONE = 0,
  CAUSAL_LM_ERROR_INVALID_PARAMETER = 1,
  CAUSAL_LM_ERROR_MODEL_LOAD_FAILED = 2,
  CAUSAL_LM_ERROR_INFERENCE_FAILED = 3,
  CAUSAL_LM_ERROR_NOT_INITIALIZED = 4,
  CAUSAL_LM_ERROR_INFERENCE_NOT_RUN = 5,
  CAUSAL_LM_ERROR_UNKNOWN = 99
} ErrorCode;

typedef enum {
  CAUSAL_LM_BACKEND_CPU = 0,
  CAUSAL_LM_BACKEND_GPU = 1,
  CAUSAL_LM_BACKEND_NPU = 2,
} BackendType;

typedef enum {
  CAUSAL_LM_MODEL_QWEN3_0_6B = 0,
  CAUSAL_LM_MODEL_GAUSS2_5 = 1,
} ModelType;

typedef struct {
  bool use_chat_template;
  bool debug_mode;
  bool verbose;
} Config;

WIN_EXPORT ErrorCode setOptions(Config config);

typedef enum {
  CAUSAL_LM_QUANTIZATION_UNKNOWN = 0,
  CAUSAL_LM_QUANTIZATION_W4A32 = 1,
  CAUSAL_LM_QUANTIZATION_W16A16 = 2,
  CAUSAL_LM_QUANTIZATION_W8A16 = 3,
  CAUSAL_LM_QUANTIZATION_W32A32 = 4,
} ModelQuantizationType;

WIN_EXPORT ErrorCode loadModel(BackendType compute, ModelType modeltype,
                               ModelQuantizationType quant_type);

typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} PerformanceMetrics;

WIN_EXPORT ErrorCode getPerformanceMetrics(PerformanceMetrics *metrics);

WIN_EXPORT ErrorCode runModel(const char *inputTextPrompt,
                              const char **outputText);

#ifdef __cplusplus
}
#endif

#endif /* __CAUSAL_LM_API_H__ */

#endif /* __QUICK_DOT_AI_API_H__ */