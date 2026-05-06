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
#include "android_memory_allocator.h"
#include "generate_qnn_utils.h"

#include <llm_util.hpp>
#include "api/streamer.h"
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

using namespace causallm;

namespace {

constexpr int kKvNumColumns = 128;

bool starts_with(const std::string &value, const std::string &prefix) {
  return value.compare(0, prefix.size(), prefix) == 0;
}

int find_tensor_index_or_minus_one(const TensorInfoList &tensor_infos,
                                   const std::string &tensor_name) {
  for (size_t idx = 0; idx < tensor_infos.size(); idx++) {
    if (tensor_infos[idx].first == tensor_name) {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

std::string kv_output_to_input_name(const std::string &output_name) {
  if (output_name.size() >= 4 &&
      output_name.compare(output_name.size() - 4, 4, "_out") == 0) {
    return output_name.substr(0, output_name.size() - 4) + "_in";
  }
  return output_name;
}

} // namespace

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
}


void causallm::Gauss3_6_QNN::initialize() {
  // Call base class initialize first - this populates models map with
  // model_inputs
  Quick_Dot_AI_QNN::initialize();
  LOGD("Quick_Dot_AI_QNN::initialize() done");

  // Get prefill and generation graph names
  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  LOGD("----------------------- initialize() %s, %s", prefill_graph.c_str(), generation_graph.c_str());
  // Get references to graph_info and model_inputs
  auto &prefill_graph_info = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;
  LOGD("----------------------- initialize() prefill_inputs : %lu", prefill_inputs.size());

  prefill_attention_mask_elements = GraphParser::get_named_tensor_elements_or_throw(
      prefill_graph_info.raw_inputs, "attention_mask");
  prefill_sliding_attention_mask_elements = GraphParser::get_named_tensor_elements_or_throw(
      prefill_graph_info.raw_inputs, "sliding_attention_mask");
  generation_attention_mask_elements = GraphParser::get_named_tensor_elements_or_throw(
      generation_graph_info.raw_inputs, "attention_mask");
  generation_sliding_attention_mask_elements =
      GraphParser::get_named_tensor_elements_or_throw(generation_graph_info.raw_inputs,
                                         "sliding_attention_mask");
  generation_full_kv_past_length = generation_attention_mask_elements - 1;
  generation_sliding_kv_past_length =
      generation_sliding_attention_mask_elements - 1;
  rope_cache_seq_len =
      std::max(max_seq_len, generation_attention_mask_elements);

  // Find input indices by name
  int prefill_input_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "inputs_embeds");
  int generation_input_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "inputs_embeds");
  LOGD("----------------------- prefill_inputs_idx : %d generation_input_idx :%d ", prefill_input_idx, generation_input_idx);
  // Save pointers to input samples
  input_sample = std::get<float *>(prefill_inputs[prefill_input_idx]);
  generation_sample = std::get<float *>(generation_inputs[generation_input_idx]);

  // Find and save pointers to other input tensors by name
  // Attention masks
  int prefill_attn_mask_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "attention_mask");
  int prefill_sliding_attn_mask_idx = GraphParser::find_tensor_index(
      prefill_graph_info.raw_inputs, "sliding_attention_mask");
  int generation_attn_mask_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "attention_mask");
  int generation_sliding_attn_mask_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "sliding_attention_mask");

  attention_mask =
      std::get<uint16_t *>(prefill_inputs[prefill_attn_mask_idx]);
  sliding_attention_mask =
      std::get<uint16_t *>(prefill_inputs[prefill_sliding_attn_mask_idx]);
  generation_attention_mask =
      std::get<uint16_t *>(generation_inputs[generation_attn_mask_idx]);
  generation_sliding_attention_mask = std::get<uint16_t *>(
      generation_inputs[generation_sliding_attn_mask_idx]);

  // Position IDs
  int prefill_pos_cos_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "position_ids_cos");
  int prefill_pos_sin_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "position_ids_sin");
  int generation_pos_cos_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "position_ids_cos");
  int generation_pos_sin_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs, "position_ids_sin");

  prefill_position_ids_cos =
      std::get<uint16_t *>(prefill_inputs[prefill_pos_cos_idx]);
  prefill_position_ids_sin =
      std::get<uint16_t *>(prefill_inputs[prefill_pos_sin_idx]);
  generation_position_ids_cos =
      std::get<uint16_t *>(generation_inputs[generation_pos_cos_idx]);
  generation_position_ids_sin =
      std::get<uint16_t *>(generation_inputs[generation_pos_sin_idx]);
  LOGD("----------------------- initialize() 1");

  // SWA Position IDs
  int prefill_swa_pos_cos_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "swa_position_ids_cos");
  int prefill_swa_pos_sin_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs, "swa_position_ids_sin");
  int generation_swa_pos_cos_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "swa_position_ids_cos");
  int generation_swa_pos_sin_idx = GraphParser::find_tensor_index(
      generation_graph_info.raw_inputs, "swa_position_ids_sin");
  LOGD("----------------------- initialize() 2");
  prefill_swa_position_ids_cos =
      std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_cos_idx]);
  prefill_swa_position_ids_sin =
      std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_sin_idx]);
  generation_swa_position_ids_cos =
      std::get<uint16_t *>(generation_inputs[generation_swa_pos_cos_idx]);
  generation_swa_position_ids_sin =
      std::get<uint16_t *>(generation_inputs[generation_swa_pos_sin_idx]);
  LOGD("----------------------- initialize() 3");

  // Allocate position_ids_cos/sin using get_cos_sin (these are source data)
  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);
  allocated_ptrs_.insert(position_ids_cos);
  allocated_ptrs_.insert(position_ids_sin);

  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, pos_dim, local_rope_theta);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  allocated_ptrs_.insert(swa_position_ids_cos);
  allocated_ptrs_.insert(swa_position_ids_sin);
  LOGD("----------------------- initialize() 4");
    // Initialize LoRA tensors
  if (lora_path.empty()) {
    // Default: fill with 32768 (zero value for quantized uint16_t)
    for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size(); idx++) {
      const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
      if (name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        auto *lora_ptr = std::get<uint16_t *>(prefill_inputs[idx]);
        std::fill_n(lora_ptr, size / sizeof(uint16_t), 32768);
      }
    }
    for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
      const auto &[name, info] = generation_graph_info.raw_inputs[idx];
      if (name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        auto *lora_ptr = std::get<uint16_t *>(generation_inputs[idx]);
        std::fill_n(lora_ptr, size / sizeof(uint16_t), 32768);
      }
    }
  LOGD("----------------------- initialize() 5");
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
      const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
      if (name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        memcpy(std::get<uint16_t *>(prefill_inputs[idx]), data_ptr, size);
        data_ptr += size;
      }
    }
  LOGD("----------------------- initialize() 6");
    // Copy to generation lora inputs (in model input order)
    for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
      const auto &[name, info] = generation_graph_info.raw_inputs[idx];
      if (name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        memcpy(std::get<uint16_t *>(generation_inputs[idx]), data_ptr, size);
        data_ptr += size;
      }
    }
  LOGD("----------------------- initialize() 7");
    munmap(mapped, file_size);
    close(fd);

    std::cout << "LoRA weights loaded from: " << lora_path << std::endl;
  }

  // Initialize KV cache with one shared backing buffer per generation KV input.
  // Prefill inputs are rebound only when that graph actually exposes the KV.
  this->fresh_kvs.clear();
  this->kvs.clear();
  this->kv_sizes.clear();
  this->kv_row_lengths.clear();
  this->prefill_output_kv_bindings.clear();
  this->generation_output_kv_bindings.clear();
  LOGD("----------------------- initialize() 8");

  std::unordered_map<std::string, int> generation_kv_index_by_name;

  for (int layer = 0; layer < num_hidden_layers; layer++) {
    const std::string key_h0_name =
        "past_key_" + std::to_string(layer) + "_h0_in";
    const auto &generation_key_h0_info =
        GraphParser::get_tensor_info_or_throw(generation_graph_info.raw_inputs,
                                              key_h0_name);
    this->kv_row_lengths.push_back(generation_key_h0_info.dimensions.back());

    const std::vector<std::string> kv_names = {
        key_h0_name,
        "past_key_" + std::to_string(layer) + "_h1_in",
        "past_value_" + std::to_string(layer) + "_h0_in",
        "past_value_" + std::to_string(layer) + "_h1_in",
    };

    for (const auto &name : kv_names) {
      int generation_input_index =
          GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                         name);
      int prefill_input_index =
          find_tensor_index_or_minus_one(prefill_graph_info.raw_inputs, name);
      const auto &generation_info =
          generation_graph_info.raw_inputs[generation_input_index].second;

      int size = GraphParser::get_tensor_size(generation_info);
      if (prefill_input_index >= 0) {
        const auto &prefill_info =
            prefill_graph_info.raw_inputs[prefill_input_index].second;
        size = std::max(size, GraphParser::get_tensor_size(prefill_info));
      }

      auto *current_kv = static_cast<uint8_t *>(tracked_allocate(size));
      auto *fresh_kv = static_cast<uint8_t *>(tracked_allocate(size));
      std::fill_n(current_kv, size, static_cast<uint8_t>(128));
      std::fill_n(fresh_kv, size, static_cast<uint8_t>(128));

      int kv_input_index = static_cast<int>(this->kvs.size());
      this->kvs.push_back(current_kv);
      this->fresh_kvs.push_back(fresh_kv);
      this->kv_sizes.push_back(size);
      generation_kv_index_by_name[name] = kv_input_index;

      if (prefill_input_index >= 0) {
        prefill_inputs[prefill_input_index] = current_kv;
      }
      generation_inputs[generation_input_index] = current_kv;
    }
  }

  auto build_output_kv_bindings =
      [&](const TensorInfoList &outputs, const std::string &graph_name) {
        std::vector<KvOutputBinding> bindings;
        for (size_t idx = 0; idx < outputs.size(); idx++) {
          const auto &name = outputs[idx].first;
          if (!starts_with(name, "past_")) {
            continue;
          }

          auto input_name = kv_output_to_input_name(name);
          auto it = generation_kv_index_by_name.find(input_name);
          if (it == generation_kv_index_by_name.end()) {
            throw std::runtime_error(graph_name +
                                     " KV output has no generation input: " +
                                     name);
          }

          int kv_index = it->second;
          bindings.push_back(
              {static_cast<int>(idx), kv_index, kv_index / 4,
               starts_with(name, "past_key_")});
        }
        return bindings;
      };

  this->prefill_output_kv_bindings =
      build_output_kv_bindings(prefill_graph_info.raw_outputs, prefill_graph);
  this->generation_output_kv_bindings = build_output_kv_bindings(
      generation_graph_info.raw_outputs, generation_graph);

  LOGD("KV cache mapping: shared_inputs=%zu prefill_outputs=%zu "
       "generation_outputs=%zu",
       this->kvs.size(), this->prefill_output_kv_bindings.size(),
       this->generation_output_kv_bindings.size());

  initialize_kv_cache();
  LOGD("----------------------- initialize() done");
}

void causallm::Gauss3_6_QNN::initialize_kv_cache() {
  kv_len = 0;
  
  // KV Cache Initialization
  for (int i = 0; i < this->kvs.size(); i++) {
    std::memcpy(this->kvs[i], this->fresh_kvs[i], this->kv_sizes[i]);
  }
}

void causallm::Gauss3_6_QNN::setupParameters(json &cfg, json &generation_cfg,
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

void causallm::Gauss3_6_QNN::run(const WSTR prompt, bool do_sample,
                                 const WSTR system_prompt,
                                 const WSTR tail_prompt, bool log_output) {
  stop_requested_.store(false, std::memory_order_release);

  auto input = tokenizer->Encode(prompt);

  if (input.size() <= 1) {
    std::cout << "[Error] Input is empty or invalid" << std::endl;
    return;
  }
  unsigned int input_len = input.size() - 1;

  auto n_chunks = (input_len % 256 != 0) ? ((input_len / 256) + 1) : (input_len / 256);
  auto token  = input.back();

  std::cout << "len: " << input_len << ", n_chunks: " << n_chunks << std::endl;

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  // Get prefill and generation graph names
  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  // Get prefill and generation models from the models map
  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  auto fill_generation_inputs = [&](int current_token, int position) {
    if (position < 0 || position >= rope_cache_seq_len) {
      throw std::runtime_error("Generation position is out of rope cache");
    }

    generation_sample[0] = current_token;

    std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
    std::fill_n(generation_sliding_attention_mask,
                generation_sliding_attention_mask_elements, 0);

    generation_attention_mask[generation_attention_mask_elements - 1] =
        std::numeric_limits<uint16_t>::max();
    generation_sliding_attention_mask
        [generation_sliding_attention_mask_elements - 1] =
            std::numeric_limits<uint16_t>::max();

    // The KV cache contains only previous tokens; the current token uses the
    // fixed final column in the generation mask.
    for (int i = 0; i < position && i < generation_full_kv_past_length; i++)
      generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
    for (int i = 0; i < position && i < generation_sliding_kv_past_length; i++)
      generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

    std::memcpy(generation_position_ids_cos,
                position_ids_cos + position * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_position_ids_sin,
                position_ids_sin + position * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_cos,
                swa_position_ids_cos + position * pos_dim,
                pos_dim * sizeof(uint16_t));
    std::memcpy(generation_swa_position_ids_sin,
                swa_position_ids_sin + position * pos_dim,
                pos_dim * sizeof(uint16_t));
  };

  auto append_outputs_to_kv_cache =
      [&](const std::vector<ml::train::TensorDim::IO_TensorType> &step_outputs,
          const std::vector<KvOutputBinding> &bindings, int target_position,
          int rows, int src_row_length, const std::string &graph_name) {
        for (const auto &binding : bindings) {
          if (binding.output_index < 0 ||
              binding.output_index >= (int)step_outputs.size() ||
              binding.kv_index < 0 || binding.kv_index >= (int)this->kvs.size() ||
              binding.layer_index < 0 ||
              binding.layer_index >= (int)this->kv_row_lengths.size()) {
            throw std::runtime_error(graph_name +
                                     " output KV binding is out of range");
          }
        }

#pragma omp parallel for
        for (int binding_idx = 0; binding_idx < (int)bindings.size();
             binding_idx++) {
          const auto &binding = bindings[binding_idx];
          int dest_row_length = kv_row_lengths[binding.layer_index];
          auto output = std::get<uint8_t *>(step_outputs[binding.output_index]);
          auto dest = this->kvs[binding.kv_index];
          int num_column = kKvNumColumns;

          int target_idx = target_position;
          int valid_before = std::min(target_position, dest_row_length);
          int shift = valid_before + rows - dest_row_length;
          if (shift > 0) {
            target_idx = valid_before - shift;
            if (binding.is_key) {
              for (int col = 0; col < num_column; ++col) {
                uint8_t *col_base = dest + col * dest_row_length;
                std::memmove(col_base, col_base + shift,
                             dest_row_length - shift);
              }
            } else {
              std::memmove(dest, dest + shift * num_column,
                           (dest_row_length - shift) * num_column);
            }
          }

          if (binding.is_key) {
            process_key(output, rows, num_column, dest, target_idx,
                        dest_row_length, src_row_length);
          } else {
            process_value(output, rows, num_column, dest, target_idx);
          }
        }
      };

  for (int c = 0; c < n_chunks; c++) {
    int chunk_len = ((c + 1) * 256 < input_len)
                        ? context_size
                        : (input_len - (c * 256));
    LOGD("kv_len : %d, chunk_len : %d", kv_len, chunk_len);

    for (int i = 0; i < context_size; i++)
      input_sample[i] = (i < chunk_len) ? input[c * 256 + i] : padding_token;

    fill_attention_mask_with_length(context_size, max_seq_len, chunk_len,
                                    attention_mask);
    fill_attention_mask_with_prev_length(
        context_size, max_seq_len, std::min(kv_len, max_seq_len - context_size),
        attention_mask);

    if (kv_len >= generation_sliding_kv_past_length) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window,
                  std::numeric_limits<uint16_t>::min());
      for (int i = 0; i < chunk_len; i++) {
        for (int j = (i + 1);
             j < (i + generation_sliding_attention_mask_elements); j++) {
          sliding_attention_mask[i * sliding_window + j] =
              std::numeric_limits<uint16_t>::max();
        }
      }
    } else {
      fill_attention_mask_with_length(context_size, sliding_window, chunk_len,
                                      sliding_attention_mask);
      fill_attention_mask_with_prev_length(context_size, sliding_window, kv_len,
                                           sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    if (kv_len + chunk_len > rope_cache_seq_len) {
      throw std::runtime_error("Prefill position is out of rope cache");
    }

    auto pos_ids_offset = kv_len * pos_dim;
    std::memcpy(prefill_position_ids_cos, position_ids_cos + pos_ids_offset,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_position_ids_sin, position_ids_sin + pos_ids_offset,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_cos,
                swa_position_ids_cos + pos_ids_offset,
                chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_sin,
                swa_position_ids_sin + pos_ids_offset,
                chunk_len * pos_dim * sizeof(uint16_t));

    auto outputs = prefill_model->inference(1, prefill_inputs);
    // Append prefill output KV into the shared prefill/generation cache.
    append_outputs_to_kv_cache(outputs, this->prefill_output_kv_bindings,
                               kv_len, chunk_len, context_size, prefill_graph);

    kv_len += chunk_len;
  }

  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = kv_len;
  for (idx = prefill_len; idx < generation_full_kv_past_length; idx++) {
    fill_generation_inputs(token, idx);

    outputs = generation_model->inference(1, generation_inputs);
    append_outputs_to_kv_cache(outputs, this->generation_output_kv_bindings,
                               idx, 1, 1, generation_graph);
    kv_len += 1;
    token = sample(std::get<uint16_t *>(outputs.back()), vocab_size,
                   input.data(), input.size(), logit_scale, logit_offset,
                   repetition_penalty, temperature, top_p, top_k);

    output.push_back(token);
    if (token == eos_token) {
      break;
    } else {
      std::string decoded= tokenizer->Decode({token});
      LOGD("%d : %s",token, decoded.c_str());
      // Stream the token if a streamer is attached
      if (streamer_) {
        if (streamer_put(streamer_, decoded.c_str()) != 0) {
          // User requested cancellation via streamer
          stop_requested_.store(true, std::memory_order_release);
          break;
        }
      } else if (log_output) {
        std::cout << decoded << std::flush;
      }
      input.push_back(token);
    }
  }

  // Notify the streamer that generation is complete
  if (streamer_) {
    streamer_end(streamer_);
  }

  has_run_ = true;

  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;
  if (log_output) {
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Generation exec_time : " << raw_exec_seconds.count()
            << ", token per second: " << (idx - input_len) / raw_exec_seconds.count()
            << ", token generation time average: "
            << raw_exec_seconds.count() / (idx - input_len) << std::endl;
  }
}
