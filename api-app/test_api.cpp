// SPDX-License-Identifier: Apache-2.0
/**
 * @file    test_api.cpp
 * @brief   Test application for src C API
 * @note    This file only includes quick_dot_ai_api.h and standard headers.
 *          It links against libquick_dot_ai_api.so.
 */

#include "quick_dot_ai_api.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

// ── ANSI color codes ─────────────────────────────────────────────────────────
namespace clr {
const char *reset = "\033[0m";
const char *bold = "\033[1m";
const char *dim = "\033[2m";
const char *cyan = "\033[36m";
const char *green = "\033[32m";
const char *yellow = "\033[33m";
const char *red = "\033[31m";
const char *magenta = "\033[35m";
const char *blue = "\033[34m";
const char *bold_cyan = "\033[1;36m";
const char *bold_green = "\033[1;32m";
const char *bold_yellow = "\033[1;33m";
const char *bold_red = "\033[1;31m";
const char *bold_magenta = "\033[1;35m";
const char *bold_blue = "\033[1;34m";
const char *bold_white = "\033[1;37m";
} // namespace clr

// ── ASCII art banner ─────────────────────────────────────────────────────────
static void print_banner() {
  std::cout << clr::bold_cyan;
  std::cout << R"(
     ____        _      _           _    ___
    / __ \      (_)    | |         / \  |_ _|
   | |  | |_   _ _  ___| | __    / _ \  | |
   | |  | | | | | |/ __| |/ /   / ___ \ | |
   | |__| | |_| | | (__|   < _ / /   \ \| |_
    \___\_\\__,_|_|\___|_|\_(_)_/     \_\___| )"
            << "\n";
  std::cout << clr::reset;
  std::cout << clr::dim << "   ─────────────────────────────────────────────\n"
            << "    On-Device LLM Inference Engine  |  API Test\n"
            << "   ─────────────────────────────────────────────" << clr::reset
            << "\n\n";
}

// ── Box-drawing helpers ──────────────────────────────────────────────────────
static void print_section(const char *title, const char *color) {
  std::cout << color << clr::bold << "┌─ " << title << " ";
  int pad = 50 - static_cast<int>(strlen(title));
  for (int i = 0; i < pad; i++)
    std::cout << "─";
  std::cout << "┐" << clr::reset << "\n";
}

static void print_section_end(const char *color) {
  std::cout << color << "└";
  for (int i = 0; i < 53; i++)
    std::cout << "─";
  std::cout << "┘" << clr::reset << "\n\n";
}

static void print_kv(const char *key, const std::string &value,
                     const char *color) {
  std::cout << color << "│" << clr::reset << "  " << clr::dim << std::left
            << std::setw(18) << key << clr::reset << clr::bold_white << value
            << clr::reset << "\n";
}

static void print_status(const char *msg, const char *icon, const char *color) {
  std::cout << color << icon << "  " << msg << clr::reset << "\n";
}

static void print_error(const std::string &msg) {
  std::cout << clr::bold_red << "  ERROR " << clr::reset << clr::red << msg
            << clr::reset << "\n";
}

// ── Usage ────────────────────────────────────────────────────────────────────
static void print_usage(const char *prog) {
  print_banner();
  print_section("Usage", clr::yellow);
  std::cout << clr::yellow << "│" << clr::reset << "  " << clr::bold_white
            << prog << clr::reset << " <model> [prompt] [chat_tpl] [quant] "
            << "[verbose]\n";
  std::cout << clr::yellow << "│" << clr::reset << "\n";
  print_kv("model", "qwen3-0.6b | gauss2.5-1b | gauss-3.6-qnn", clr::yellow);
  print_kv("prompt", "\"Hello, how are you?\"", clr::yellow);
  print_kv("chat_tpl", "true | false  (default: true)", clr::yellow);
  print_kv("quant", "W4A32 | W16A16 | W8A16 | W32A32", clr::yellow);
  print_kv("verbose", "true | false  (default: true)", clr::yellow);
  print_kv("model_base_path",
           "Base directory for models (or set QUICKAI_MODEL_BASE_PATH)",
           clr::yellow);
  print_section_end(clr::yellow);
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_error("Missing required argument: <model>");
    print_usage(argv[0]);
    return 1;
  }

  // ── Parse arguments ──────────────────────────────────────────────────────
  const char *model_name = argv[1];
  const char *prompt = (argc >= 3) ? argv[2] : "Hello, how are you?";

  bool use_chat_template = true;
  if (argc >= 4) {
    std::string arg(argv[3]);
    std::transform(arg.begin(), arg.end(), arg.begin(), ::tolower);
    use_chat_template = (arg == "true" || arg == "1");
  }

  ModelQuantizationType quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
  std::string quant_str = "W4A32";
  if (argc >= 5) {
    quant_str = std::string(argv[4]);
    if (quant_str == "W4A32") {
      quant_type = CAUSAL_LM_QUANTIZATION_W4A32;
    } else if (quant_str == "W16A16") {
      quant_type = CAUSAL_LM_QUANTIZATION_W16A16;
    } else if (quant_str == "W8A16") {
      quant_type = CAUSAL_LM_QUANTIZATION_W8A16;
    } else if (quant_str == "W32A32") {
      quant_type = CAUSAL_LM_QUANTIZATION_W32A32;
    }
  }

  bool verbose = true;
  if (argc >= 6) {
    std::string arg(argv[5]);
    verbose = (arg == "1" || arg == "true");
  }

  // Model base path: CLI arg > env var > nullptr (uses C API default)
  const char *model_base_path = nullptr;
  std::string model_base_path_storage;
  if (argc >= 7) {
    model_base_path_storage = argv[6];
    model_base_path = model_base_path_storage.c_str();
  } else {
    const char *env_path = std::getenv("QUICKAI_MODEL_BASE_PATH");
    if (env_path != nullptr && strlen(env_path) > 0) {
      model_base_path_storage = env_path;
      model_base_path = model_base_path_storage.c_str();
    }
  }

  // ── Banner ─────────────────────────────────────────────────────────────
  print_banner();

  // ── Configuration ──────────────────────────────────────────────────────
  print_section("Configuration", clr::cyan);
  print_kv("Model", model_name, clr::cyan);
  print_kv("Prompt", std::string("\"") + prompt + "\"", clr::cyan);
  print_kv("Chat Template", use_chat_template ? "Yes" : "No", clr::cyan);
  print_kv("Quantization", quant_str, clr::cyan);
  print_kv("Verbose", verbose ? "Yes" : "No", clr::cyan);
  print_kv("Model Base Path",
           model_base_path ? model_base_path : "(C API default)", clr::cyan);
  print_section_end(clr::cyan);

  // ── Set options ────────────────────────────────────────────────────────
  Config config;
  config.use_chat_template = use_chat_template;
  config.debug_mode = verbose;
  config.verbose = verbose;

  ErrorCode err = setOptions(config);
  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("Failed to set options (code " + std::to_string(err) + ")");
    return 1;
  }

  // ── Resolve model type ─────────────────────────────────────────────────
  // Public models use the compat enum path (loadModelHandle).
  // Gauss models are no longer in the enum — they use loadModelHandleByName
  // via catalog id.
  std::string model_name_str(model_name);
  std::transform(model_name_str.begin(), model_name_str.end(),
                 model_name_str.begin(), ::tolower);

  ModelType model_type =
    CAUSAL_LM_MODEL_QWEN3_0_6B; // default, overridden below
  std::string catalog_id;       // non-empty → use loadModelHandleByName
  bool use_by_name = false;

  if (model_name_str == "qwen3-0.6b") {
    model_type = CAUSAL_LM_MODEL_QWEN3_0_6B;
  } else if (model_name_str == "qwen3-1.7b-q40" ||
             model_name_str == "qwen3_1.7b_q40") {
    model_type = CAUSAL_LM_MODEL_QWEN3_1_7B_Q40;
  } else if (model_name_str == "tiny_bert" || model_name_str == "tiny-bert") {
    model_type = CAUSAL_LM_MODEL_TINY_BERT;
  } else if (model_name_str == "function_gemma" ||
             model_name_str == "function-gemma") {
    model_type = CAUSAL_LM_MODEL_FUNCTION_GEMMA;
  } else if (model_name_str == "gemma4_cpu" || model_name_str == "gemma4-cpu") {
    model_type = CAUSAL_LM_MODEL_GEMMA4_CPU;
  } else if (model_name_str == "gemma4_e2b_qnn" ||
             model_name_str == "gemma4-e2b-qnn") {
    model_type = CAUSAL_LM_MODEL_GEMMA4_E2B_QNN;
  } else if (model_name_str == "gauss2.5" || model_name_str == "gauss2.5-1b") {
    // Gauss models use loadModelHandleByName (catalog-registered, no enum)
    catalog_id = "gauss2.5-1b";
    use_by_name = true;
  } else if (model_name_str == "gauss-3.6-qnn" ||
             model_name_str == "gauss-3.6_qnn") {
#ifdef ENABLE_QNN
    catalog_id = "gauss-3.6-qnn";
    use_by_name = true;
#else
    print_error("Model '" + std::string(model_name) +
                "' requires QNN support. Rebuild with -Denable-qnn=true.");
    return 1;
#endif
  } else if (model_name_str == "gauss-3.8-qnn" ||
             model_name_str == "gauss-3.8_qnn") {
#ifdef ENABLE_QNN
    catalog_id = "gauss-3.8-qnn";
    use_by_name = true;
#else
    print_error("Model '" + std::string(model_name) +
                "' requires QNN support. Rebuild with -Denable-qnn=true.");
    return 1;
#endif
  } else {
    print_error("Unknown model: " + std::string(model_name));
    return 1;
  }

  // ── Load/Unload Stress Test ────────────────────────────────────────────
  const int STRESS_CYCLES = 1;
  print_section("Load/Unload Stress Test", clr::blue);

  for (int i = 0; i < STRESS_CYCLES; ++i) {
    std::cout << clr::blue << "│" << clr::reset << "  " << clr::bold_white
              << "Cycle " << (i + 1) << "/" << STRESS_CYCLES << clr::reset
              << ": ";

    // Load
    CausalLmHandle cycle_handle = nullptr;
    if (use_by_name) {
      err = loadModelHandleByName(CAUSAL_LM_BACKEND_CPU, catalog_id.c_str(),
                                  quant_type, nullptr, model_base_path,
                                  &cycle_handle);
    } else {
      err = loadModelHandle(CAUSAL_LM_BACKEND_CPU, model_type, quant_type,
                            nullptr, model_base_path, &cycle_handle);
    }
    if (err != CAUSAL_LM_ERROR_NONE) {
      print_error("loadModel failed at cycle " + std::to_string(i + 1) +
                  " (code " + std::to_string(err) + ")");
      return 1;
    }
    std::cout << clr::bold_green << "LOAD OK" << clr::reset;

    // Unload
    err = unloadModelHandle(cycle_handle);
    if (err != CAUSAL_LM_ERROR_NONE) {
      print_error("unloadModelHandle failed at cycle " + std::to_string(i + 1) +
                  " (code " + std::to_string(err) + ")");
      destroyModelHandle(cycle_handle);
      return 1;
    }
    std::cout << clr::dim << " → " << clr::reset;
    std::cout << clr::bold_yellow << "UNLOAD OK" << clr::reset;

    // Destroy handle (unload keeps struct alive, must destroy to avoid leak)
    destroyModelHandle(cycle_handle);
    std::cout << clr::dim << " → " << clr::reset;
    std::cout << clr::dim << "DESTROY OK" << clr::reset << "\n";
  }

  std::cout << clr::blue << "│" << clr::reset << "\n";
  std::cout << clr::blue << "│" << clr::reset << "  " << clr::bold_white
            << "Final load (#" << (STRESS_CYCLES + 1) << "):" << clr::reset
            << " Loading " << clr::bold_white << model_name << clr::reset
            << " (" << quant_str << ") ...\n";

  CausalLmHandle handle = nullptr;
  if (use_by_name) {
    err = loadModelHandleByName(CAUSAL_LM_BACKEND_CPU, catalog_id.c_str(),
                                quant_type, nullptr, model_base_path, &handle);
  } else {
    err = loadModelHandle(CAUSAL_LM_BACKEND_CPU, model_type, quant_type,
                          nullptr, model_base_path, &handle);
  }
  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("Final loadModel failed (code " + std::to_string(err) + ")");
    return 1;
  }

  std::cout << clr::blue << "│" << clr::reset << "  ";
  print_status("Model loaded successfully", ">>", clr::bold_green);
  print_section_end(clr::blue);

  // ── Inference ──────────────────────────────────────────────────────────
  print_section("Inference", clr::green);
  std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
            << "Input: " << clr::reset << clr::bold_white << prompt
            << clr::reset << "\n";
  std::cout << clr::green << "│" << clr::reset << "\n";

  const char *outputText = nullptr;
  // CausalLMChatMessage msg;
  // msg.role = "user";
  // msg.content = prompt;
  // err = runModelHandleWithMessages(handle, &msg, 1, true, &outputText);

  // XGrammar Test
  auto tool_name = "web_search";
  auto schema =
    "{\"type\": \"object\",\"properties\": {\"query\": {\"type\": \"string\", "
    "\"description\": \"Search query in the most effective language for "
    "results (use Korean for Korean local info, English for global "
    "topics)\"},\"count\": {\"type\": \"integer\", \"description\": \"Number "
    "of results to return (default 5, max 10)\"}},\"required\": [\"query\"]}";
  err = runModelHandleWithTool(handle, prompt, &outputText, tool_name, schema);

  if (err != CAUSAL_LM_ERROR_NONE) {
    print_error("Inference failed (code " + std::to_string(err) + ")");
    return 1;
  }

  if (outputText) {
    std::cout << clr::green << "│" << clr::reset << "  " << clr::dim
              << "Output:" << clr::reset << "\n";
    std::cout << clr::green << "│" << clr::reset << "  " << clr::bold_white
              << outputText << clr::reset << "\n";
  }
  print_section_end(clr::green);

  // ── Performance Metrics ────────────────────────────────────────────────
  print_section("Performance Metrics", clr::magenta);

  PerformanceMetrics metrics;
  memset(&metrics, 0, sizeof(metrics));
  err = getPerformanceMetricsHandle(handle, &metrics);
  if (err == CAUSAL_LM_ERROR_NONE) {
    double prefill_tps =
      metrics.prefill_duration_ms > 0
        ? metrics.prefill_tokens / metrics.prefill_duration_ms * 1000.0
        : 0.0;
    double gen_tps =
      metrics.generation_duration_ms > 0
        ? metrics.generation_tokens / metrics.generation_duration_ms * 1000.0
        : 0.0;

    std::ostringstream oss;

    oss << metrics.initialization_duration_ms << " ms";
    print_kv("Init", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.prefill_tokens << " tokens / " << metrics.prefill_duration_ms
        << " ms  (" << std::fixed << std::setprecision(1) << prefill_tps
        << " tok/s)";
    print_kv("Prefill", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.generation_tokens << " tokens / "
        << metrics.generation_duration_ms << " ms  (" << std::fixed
        << std::setprecision(1) << gen_tps << " tok/s)";
    print_kv("Generation", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.total_duration_ms << " ms";
    print_kv("Total", oss.str(), clr::magenta);

    oss.str("");
    oss << metrics.peak_memory_kb << " KB";
    print_kv("Peak Memory", oss.str(), clr::magenta);

    // ── Metric validation ────────────────────────────────────────────────
    bool metrics_ok = true;
    if (metrics.prefill_tokens == 0) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: prefill_tokens is zero" << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics.generation_tokens == 0) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: generation_tokens is zero" << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics.generation_duration_ms <= 0) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: generation_duration_ms is zero/negative"
                << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics.total_duration_ms <
        metrics.prefill_duration_ms + metrics.generation_duration_ms - 0.001) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::yellow
                << "⚠ Warning: total_duration_ms < prefill + generation"
                << clr::reset << "\n";
      metrics_ok = false;
    }
    if (metrics_ok) {
      std::cout << clr::magenta << "│" << clr::reset << "  " << clr::green
                << "✓ All metric sanity checks passed" << clr::reset << "\n";
    }
  } else {
    std::cout << clr::magenta << "│" << clr::reset << "  " << clr::dim
              << "(metrics not available)" << clr::reset << "\n";
  }
  print_section_end(clr::magenta);

  // ── Cleanup ────────────────────────────────────────────────────────────
  destroyModelHandle(handle);

  // ── Done ───────────────────────────────────────────────────────────────
  std::cout << clr::bold_green << "  Done." << clr::reset << "\n\n";

  return 0;
}
