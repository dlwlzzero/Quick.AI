// SPDX-License-Identifier: Apache-2.0
/**
 * @file   Gauss3_8_VIT_QNN.cpp
 * @brief  QNN model implementation with self-registration
 * @note   This model auto-registers with the CausalLM Factory via
 * __attribute__((constructor)) when linked or loaded.
 *
 *         No modification to nntrainer's main.cpp is needed.
 */

#include "gauss3_8_vit_qnn.h"
#include "android_memory_allocator.h"
#include "generate_qnn_utils.h"

#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>
#include <model_descriptor.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace causallm;

/**
 * @brief Auto-registration via constructor attribute
 *
 * This function runs automatically when the shared library is loaded
 * (before main()). It registers all custom models with the CausalLM Factory.
 *
 */
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
    "Gauss_3_8_Visual_VIT_QNN",
    [](causallm::json cfg, causallm::json generation_cfg,
       causallm::json nntr_cfg) {
      return std::make_unique<causallm::Gauss3_8_VIT_QNN>(cfg, generation_cfg,
                                                          nntr_cfg);
    });

  // Composite preset: catalog id "gauss-3.8-vit-qnn" resolves to the device
  // dir whose nntr_config.json composes [Gauss_3_8_VEncoder_QNN,
  // Gauss_3_8_QNN]. The generic api multimodal path then drives the pair.
  static const ModelDescriptor d = {"gauss-3.8-vit-qnn",
                                    "gauss-3.8-vision",
                                    "Gauss 3.8 Vision (QNN)",
                                    QDA_RUNTIME_NATIVE,
                                    (1u << 2),
                                    QDA_CAP_MULTIMODAL | QDA_CAP_MESSAGES_API,
                                    "gauss-3.8-vit-qnn",
                                    "Gauss_3_8_VEncoder_QNN"};
  quick_dot_ai::register_model_descriptor(&d);
}

void causallm::Gauss3_8_VIT_QNN::initialize() {
  // Call base class initialize first - this populates models map with
  // model_inputs
  Quick_Dot_AI_QNN::initialize();

  // Get prefill and generation graph names
  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  // Get references to graph_info and model_inputs
  auto &prefill_graph_info = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  // Find input indices by name
  int prefill_input_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "inputs_embeds");
  int generation_input_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "inputs_embeds");

  // Save pointers to input samples
  input_sample = std::get<float *>(prefill_inputs[prefill_input_idx]);
  generation_sample =
    std::get<float *>(generation_inputs[generation_input_idx]);

  // Find and save pointers to other input tensors by name
  // Attention masks
  int prefill_attn_mask_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "attention_mask");
  int prefill_sliding_attn_mask_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "sliding_attention_mask");
  int generation_attn_mask_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "attention_mask");
  int generation_sliding_attn_mask_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "sliding_attention_mask");

  attention_mask = std::get<uint16_t *>(prefill_inputs[prefill_attn_mask_idx]);
  sliding_attention_mask =
    std::get<uint16_t *>(prefill_inputs[prefill_sliding_attn_mask_idx]);
  generation_attention_mask =
    std::get<uint16_t *>(generation_inputs[generation_attn_mask_idx]);
  generation_sliding_attention_mask =
    std::get<uint16_t *>(generation_inputs[generation_sliding_attn_mask_idx]);

  // Position IDs
  int prefill_pos_cos_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "position_ids_cos");
  int prefill_pos_sin_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "position_ids_sin");
  int generation_pos_cos_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "position_ids_cos");
  int generation_pos_sin_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "position_ids_sin");

  prefill_position_ids_cos =
    std::get<uint16_t *>(prefill_inputs[prefill_pos_cos_idx]);
  prefill_position_ids_sin =
    std::get<uint16_t *>(prefill_inputs[prefill_pos_sin_idx]);
  generation_position_ids_cos =
    std::get<uint16_t *>(generation_inputs[generation_pos_cos_idx]);
  generation_position_ids_sin =
    std::get<uint16_t *>(generation_inputs[generation_pos_sin_idx]);

  // SWA Position IDs
  int prefill_swa_pos_cos_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "swa_position_ids_cos");
  int prefill_swa_pos_sin_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "swa_position_ids_sin");
  int generation_swa_pos_cos_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "swa_position_ids_cos");
  int generation_swa_pos_sin_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "swa_position_ids_sin");

  prefill_swa_position_ids_cos =
    std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_cos_idx]);
  prefill_swa_position_ids_sin =
    std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_sin_idx]);
  generation_swa_position_ids_cos =
    std::get<uint16_t *>(generation_inputs[generation_swa_pos_cos_idx]);
  generation_swa_position_ids_sin =
    std::get<uint16_t *>(generation_inputs[generation_swa_pos_sin_idx]);

  // Allocate position_ids_cos/sin using get_cos_sin (these are source data)
  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
    get_cos_sin(context_size, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);

  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
    get_cos_sin(context_size, pos_dim, local_rope_theta);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);

  // Initialize LoRA tensors
  if (lora_path.empty()) {
    // Default: fill with 32768 (zero value for quantized uint16_t)
    for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size(); idx++) {
      const auto &info = prefill_graph_info.raw_inputs[idx];
      if (info.name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        auto *lora_ptr = std::get<uint16_t *>(prefill_inputs[idx]);
        std::fill_n(lora_ptr, size / sizeof(uint16_t), 32768);
      }
    }
    for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
      const auto &info = generation_graph_info.raw_inputs[idx];
      if (info.name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        auto *lora_ptr = std::get<uint16_t *>(generation_inputs[idx]);
        std::fill_n(lora_ptr, size / sizeof(uint16_t), 32768);
      }
    }
  } else {
    // Load from lora_path file
    int fd = open(lora_path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("Failed to open lora_path: " + lora_path);
    }

    struct stat st;
    if (fstat(fd, &st) < 0) {
      close(fd);
      throw std::runtime_error("Failed to stat lora_path: " + lora_path);
    }
    size_t file_size = st.st_size;

    void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to mmap lora_path: " + lora_path);
    }

    uint8_t *data_ptr = static_cast<uint8_t *>(mapped);

    // Copy to prefill lora inputs (in model input order)
    for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size(); idx++) {
      const auto &info = prefill_graph_info.raw_inputs[idx];
      if (info.name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        memcpy(std::get<uint16_t *>(prefill_inputs[idx]), data_ptr, size);
        data_ptr += size;
      }
    }

    // Copy to generation lora inputs (in model input order)
    for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
      const auto &info = generation_graph_info.raw_inputs[idx];
      if (info.name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        memcpy(std::get<uint16_t *>(generation_inputs[idx]), data_ptr, size);
        data_ptr += size;
      }
    }

    munmap(mapped, file_size);
    close(fd);

    std::cout << "LoRA weights loaded from: " << lora_path << std::endl;
  }

  // Initialize KV cache - find tensors starting with "past_" from model_inputs
  this->fresh_kvs.clear();
  this->kvs.clear();
  this->kv_sizes.clear();

  // Find all KV cache tensors (names starting with "past_") in generation
  // inputs
  for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
    const auto &info = generation_graph_info.raw_inputs[idx];
    if (info.name.find("past_") == 0) {
      // Found a KV cache tensor
      auto *kv_ptr = std::get<uint8_t *>(generation_inputs[idx]);
      this->kvs.push_back((uint16_t *)kv_ptr);

      // Calculate size using GraphParser::get_tensor_size
      int size = GraphParser::get_tensor_size(info);
      this->kv_sizes.push_back(size);

      // Allocate fresh_kvs for reset during run()
      this->fresh_kvs.push_back(
        (uint16_t *)get_zero_memory(size, 128 * 256 + 128));
    }
  }
}

void causallm::Gauss3_8_VIT_QNN::setupParameters(json &cfg,
                                                 json &generation_cfg,
                                                 json &nntr_cfg) {
  // Call base class setupParameters first
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  // Read config parameters - model dimensions
  num_hidden_layers = cfg["num_hidden_layers"].get<int>();
  max_window_layers = cfg["max_window_layers"].get<int>();
  hidden_size = cfg["hidden_size"].get<int>();
  sequence_length = cfg["sequence_length"].get<int>();
  vocab_size = cfg["vocab_size"].get<int>();
  max_seq_len = cfg["max_seq_len"].get<int>();
  sliding_window = cfg["sliding_window"].get<int>();
  local_rope_theta = cfg["local_rope_theta"].get<float>();
  rope_theta = cfg["rope_theta"].get<float>();
  context_size = cfg["context_size"].get<int>();
  pos_dim = cfg["pos_dim"].get<int>();
  head_dim = cfg["head_dim"].get<int>();

  // Read generation_config parameters
  padding_token = generation_cfg["padding_token"].get<int>();
  eos_token = generation_cfg["eos_token_id"].get<int>();
  temperature = generation_cfg["temperature"].get<float>();
  top_k = generation_cfg["top_k"].get<int>();
  top_p = generation_cfg["top_p"].get<float>();
  repetition_penalty = generation_cfg["repetition_penalty"].get<float>();
  logit_scale = generation_cfg["logit_scale"].get<float>();
  logit_offset = generation_cfg["logit_offset"].get<int>();

  // Read optional lora_path
  lora_path = nntr_cfg.value("lora_path", "");
}

void causallm::Gauss3_8_VIT_QNN::run(const WSTR prompt, bool do_sample,
                                     const WSTR system_prompt,
                                     const WSTR tail_prompt, bool log_output) {
  // Get prefill and generation graph names
  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  // auto _input = tokenizer->Encode(prompt);
  // unsigned int _len = _input.size() - 1;
  // for (int i = 0; i < context_size; i++)
  //   input_sample[i] = padding_token;
  // std::cout << "input length: " << _len << std::endl;
  // for (int i = 0; i < _len; i++) {
  //   std::cout << _input[i] << ", ";
  //   input_sample[i] = _input[i];
  // }
  // std::cout << _input[_len] << std::endl;

  // Actual prompt tokenization for metrics; vision embedding is injected
  // externally by the caller before this model runs.
  auto _input = tokenizer->Encode(promptToUtf8(prompt));
  unsigned int _len = static_cast<unsigned int>(_input.size());
  if (_len == 0) {
    _len = 1;
    _input = {1};
  }

  fill_attention_mask_with_length(context_size, max_seq_len, _len,
                                  attention_mask);
  fill_attention_mask_with_length(context_size, sliding_window, _len,
                                  sliding_attention_mask);
  std::fill_n(generation_attention_mask, max_seq_len, 0);
  std::fill_n(generation_sliding_attention_mask, sliding_window - context_size,
              0);
  generation_attention_mask[max_seq_len - 1] =
    std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < _len; i++)
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask[(sliding_window - context_size) - 1] =
    std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < _len; i++)
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
  std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
  std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
  std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);
  std::memcpy(prefill_position_ids_cos, position_ids_cos,
              _len * pos_dim * sizeof(uint16_t));
  std::memcpy(prefill_position_ids_sin, position_ids_sin,
              _len * pos_dim * sizeof(uint16_t));
  std::memcpy(prefill_swa_position_ids_cos, swa_position_ids_cos,
              _len * pos_dim * sizeof(uint16_t));
  std::memcpy(prefill_swa_position_ids_sin, swa_position_ids_sin,
              _len * pos_dim * sizeof(uint16_t));

  for (int i = 0; i < this->kvs.size(); i++) {
    std::memcpy(this->kvs[i], this->fresh_kvs[i], this->kv_sizes[i]);
  }

  // Get prefill and generation models from the models map
  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  std::cout << "before prefill model run..." << std::endl;
  auto prefill_start = std::chrono::high_resolution_clock::now();
  auto outputs = prefill_model->inference(1, prefill_inputs);
  auto prefill_end = std::chrono::high_resolution_clock::now();
  auto token = _input.back();
  std::vector<int> output;

  auto start = std::chrono::system_clock::now();
  int idx;
  for (idx = _len; idx < context_size; idx++) {
    generation_sample[0] = token;

    generation_attention_mask[idx] = std::numeric_limits<uint16_t>::max();
    generation_sliding_attention_mask[idx] =
      std::numeric_limits<uint16_t>::max();
    std::memcpy(generation_position_ids_cos, position_ids_cos + idx * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_position_ids_sin, position_ids_sin + idx * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_cos,
                swa_position_ids_cos + idx * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_sin,
                swa_position_ids_sin + idx * pos_dim,
                pos_dim * sizeof(uint16_t));

    // Remove output_hidden_states from outputs
    outputs.erase(outputs.begin() + 96);

#pragma omp parallel for
    for (int i = 0; i < this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      int dest_row_length = layer_idx % 5 == 4 ? max_seq_len - context_size
                                               : sliding_window - context_size;
      int src_row_length = idx == _len ? 256 : 1;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)this->kvs[kv_idx];
      // key cache or value cache
      int num_row = idx == _len ? _len : 1;
      int num_column = 128;
      if (is_key) {
        // format: 1:1:col:row
        process_key(output, num_row, num_column, dest, idx == _len ? 0 : idx,
                    dest_row_length, src_row_length);
      } else {
        // format: 1:1:row:col
        process_value(output, num_row, num_column, dest, idx == _len ? 0 : idx);
      }
    };

    outputs = generation_model->inference(1, generation_inputs);
    token = sample(std::get<uint16_t *>(outputs.back()), vocab_size,
                   _input.data(), _input.size(), logit_scale, logit_offset,
                   repetition_penalty, temperature, top_p, top_k);

    output.push_back(token);
    if (token == eos_token || token == padding_token) {
      std::cout << "Finished generating, break..." << std::endl;
      break;
    } else {
      std::cout << tokenizer->Decode({token}) << std::flush;
      _input.push_back(token);
    }
  }
  auto end = std::chrono::system_clock::now();
  this->raw_exec_seconds = end - start;

  performance_metrics.prefill_tokens = _len;
  performance_metrics.prefill_duration_ms =
    std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
      .count();
  performance_metrics.generation_tokens =
    (idx > static_cast<int>(_len))
      ? (unsigned int)(idx - static_cast<int>(_len))
      : 0U;
  performance_metrics.generation_duration_ms =
    std::chrono::duration<double, std::milli>(end - start).count();
  performance_metrics.total_duration_ms =
    performance_metrics.prefill_duration_ms +
    performance_metrics.generation_duration_ms;
  performance_metrics.peak_memory_kb = getPeakMemoryKb();
  has_run_ = true;

  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Generation exec_time : " << this->raw_exec_seconds.count()
            << ", token per second: "
            << (idx - _len) / this->raw_exec_seconds.count()
            << ", token generation time average: "
            << this->raw_exec_seconds.count() / (idx - _len) << std::endl;
}

// ---------------------------------------------------------------------------
// Multimodal composition is handled generically in the api layer
// (execute_multimodal), driving any [vision producer, LLM consumer] pair
// through base Transformer virtuals. No per-model multimodal logic lives here.
// ---------------------------------------------------------------------------
