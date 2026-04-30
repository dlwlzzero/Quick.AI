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
#include <cmath>
#include <iostream>

using namespace causallm;

namespace {

constexpr int kEosDebugSteps = 8;

struct CandidateLogit {
  int token_id;
  float logit;
  float probability;
};

float dequantize_logit(uint16_t raw_logit, float logit_scale, int logit_offset,
                       float temperature) {
  float logit = (static_cast<float>(raw_logit) + logit_offset) * logit_scale;
  if (temperature > 1e-5f) {
    logit /= temperature;
  }
  return logit;
}

bool token_seen_in_history(int token_id, const int *tokens,
                           int number_of_tokens) {
  for (int i = 0; i < number_of_tokens; ++i) {
    if (tokens[i] == token_id) {
      return true;
    }
  }
  return false;
}

int sample_gauss3_6_corrected(uint16_t *pointer, int length, int *tokens,
                              int number_of_tokens, float logit_scale,
                              int logit_offset, float repetition_penalty,
                              float temperature, float top_p, int top_k) {
  if (length <= 0) {
    return 0;
  }

  if (top_k <= 0 || top_k > length) {
    top_k = length;
  }

  std::priority_queue<std::pair<int, int>, std::vector<std::pair<int, int>>,
                      std::greater<std::pair<int, int>>>
      top_k_elements;
  for (int i = 0; i < top_k; ++i) {
    top_k_elements.push(std::make_pair(pointer[i], i));
  }
  for (int i = top_k; i < length; ++i) {
    if (top_k_elements.top().first < pointer[i]) {
      top_k_elements.pop();
      top_k_elements.push(std::make_pair(pointer[i], i));
    }
  }

  std::vector<CandidateLogit> candidates(top_k_elements.size());
  for (int i = static_cast<int>(candidates.size()) - 1; i >= 0; --i) {
    auto element = top_k_elements.top();
    top_k_elements.pop();
    candidates[i].token_id = element.second;
    candidates[i].logit = dequantize_logit(element.first, logit_scale,
                                           logit_offset, temperature);
    candidates[i].probability = 0.0f;
  }

  if (repetition_penalty > 1e-5f && repetition_penalty != 1.0f) {
    for (auto &candidate : candidates) {
      if (token_seen_in_history(candidate.token_id, tokens, number_of_tokens)) {
        candidate.logit /= repetition_penalty;
      }
    }
  }

  float max_logit = candidates.front().logit;
  for (const auto &candidate : candidates) {
    max_logit = std::max(max_logit, candidate.logit);
  }

  float sum_exp_logits = 0.0f;
  for (auto &candidate : candidates) {
    candidate.probability = std::exp(candidate.logit - max_logit);
    sum_exp_logits += candidate.probability;
  }

  if (sum_exp_logits <= 0.0f) {
    return candidates.front().token_id;
  }

  for (auto &candidate : candidates) {
    candidate.probability /= sum_exp_logits;
  }

  std::sort(candidates.begin(), candidates.end(),
            [](const CandidateLogit &a, const CandidateLogit &b) {
              return a.probability > b.probability;
            });

  float cumulative_probability = 0.0f;
  size_t nucleus_size = 0;
  while (nucleus_size < candidates.size() &&
         (cumulative_probability < top_p || nucleus_size == 0)) {
    cumulative_probability += candidates[nucleus_size].probability;
    nucleus_size++;
  }

  std::vector<double> sampling_weights;
  sampling_weights.reserve(nucleus_size);
  for (size_t i = 0; i < nucleus_size; ++i) {
    sampling_weights.push_back(candidates[i].probability);
  }

  std::discrete_distribution<int> dist(sampling_weights.begin(),
                                       sampling_weights.end());
  return candidates[dist(rng)].token_id;
}

int select_greedy_token(uint16_t *pointer, int length) {
  return static_cast<int>(
      std::max_element(pointer, pointer + length) - pointer);
}

void log_eos_debug(uint16_t *pointer, int length, int eos_token,
                   float logit_scale, int logit_offset, float temperature,
                   int generation_step) {
  if (generation_step >= kEosDebugSteps || eos_token < 0 || eos_token >= length) {
    return;
  }

  int top_token = select_greedy_token(pointer, length);
  int eos_rank = 1;
  for (int i = 0; i < length; ++i) {
    if (pointer[i] > pointer[eos_token]) {
      eos_rank++;
    }
  }

  const float top_logit =
      dequantize_logit(pointer[top_token], logit_scale, logit_offset,
                       temperature);
  const float eos_logit =
      dequantize_logit(pointer[eos_token], logit_scale, logit_offset,
                       temperature);

  std::cout << "[EOS_DEBUG] step=" << generation_step << " top_token="
            << top_token << " eos_token=" << eos_token
            << " eos_rank=" << eos_rank << " top_logit=" << top_logit
            << " eos_logit=" << eos_logit << std::endl;
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

  // Initialize KV cache - find tensors starting with "past_" from model_inputs
  this->fresh_kvs.clear();
  this->kvs.clear();
  this->kv_sizes.clear();
  this->kv_row_lengths.clear();
  LOGD("----------------------- initialize() 8");
  // Find all KV cache tensors (names starting with "past_") in generation
  // inputs
  for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
    const auto &[name, info] = generation_graph_info.raw_inputs[idx];
    if (name.find("past_") == 0) {
      // Found a KV cache tensor
      auto *kv_ptr = std::get<uint8_t *>(generation_inputs[idx]);
      size_t kv_input_index = this->kvs.size();
      this->kvs.push_back((uint16_t *)kv_ptr);

      // Calculate size using GraphParser::get_tensor_size
      int size = GraphParser::get_tensor_size(info);
      this->kv_sizes.push_back(size);

      if (kv_input_index % 4 == 0) {
        this->kv_row_lengths.push_back(info.dimensions.back());
      }

      // Allocate fresh_kvs for reset during run()
      auto fresh_kv = (uint16_t *)get_zero_memory(size, 128 * 256 + 128);
      allocated_ptrs_.insert(fresh_kv);
      this->fresh_kvs.push_back(fresh_kv);
    }
  }

  for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size(); idx++) {
    const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
    if (name.find("past_") == 0) {
      // Found a KV cache tensor
      auto *kv_ptr = std::get<uint8_t *>(prefill_inputs[idx]);
      int size = GraphParser::get_tensor_size(info);
      std::fill_n(kv_ptr, size, 128);
    }
  }
  LOGD("----------------------- initialize() done");
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
  last_output_.clear();

  // Always start with a clean cancellation state — the streamer (if
  // any) may have flipped this flag on a previous run that was
  // cancelled, and we don't want stale state to break an unrelated
  // subsequent run().
  stop_requested_.store(false, std::memory_order_release);

  // Get prefill and generation graph names
  std::string prefill_graph = graphs_to_use[0];
  std::string generation_graph = graphs_to_use[1];

  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

  // Get prefill and generation models from the models map
  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  // KV Cache Initialization
  for (int i = 0; i < this->kvs.size(); i++) {
    std::memcpy(this->kvs[i], this->fresh_kvs[i], this->kv_sizes[i]);
  }

  auto _input = tokenizer->Encode(prompt);
  auto token  = _input.back();

  unsigned int _len = _input.size() - 1;
  if(_len <= 0){
    std::cout << "[Error] Input is empty or invalid" << std::endl;
    return;
  }

  auto _n_chunks = (_len % 256 != 0) ? ((_len / 256) + 1) : (_len / 256);

  std::cout << "n_chunk: " << _n_chunks << ", len: " << _len << std::endl;

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  for(int c = 0; c < _n_chunks; c++) {
    int _chunk_len = ((c + 1) * 256 < _len) ? context_size : (_len - (c * 256));
    int kv_len = c * context_size;

    std::cout << "chunk_num: " << c << ", chunk_len: " << _chunk_len << std::endl;

    for(int i = 0; i < context_size; i++)
      input_sample[i] = (i < _chunk_len) ? _input[c * 256 + i] : padding_token;

    fill_attention_mask_with_length(context_size, max_seq_len, _chunk_len, attention_mask);
    fill_attention_mask_with_prev_length(context_size, max_seq_len, c * 256, attention_mask);

    if (kv_len >= generation_sliding_kv_past_length) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window, std::numeric_limits<uint16_t>::min());
      for(int i = 0; i < _chunk_len; i++) {
        for(int j = (i + 1);
            j < (i + generation_sliding_attention_mask_elements); j++) {
          sliding_attention_mask[i * sliding_window + j] = std::numeric_limits<uint16_t>::max();
        }
      }
    } else {
      fill_attention_mask_with_length(context_size, sliding_window, _chunk_len, sliding_attention_mask);
      fill_attention_mask_with_prev_length(context_size, sliding_window, kv_len, sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    auto pos_ids_offset = c * context_size * pos_dim;
    std::memcpy(prefill_position_ids_cos, position_ids_cos + pos_ids_offset, _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_position_ids_sin, position_ids_sin + pos_ids_offset, _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_cos, swa_position_ids_cos + pos_ids_offset, _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_sin, swa_position_ids_sin + pos_ids_offset, _chunk_len * pos_dim * sizeof(uint16_t));

    auto outputs = prefill_model->inference(1, prefill_inputs);

    // KV Cache Copy
#pragma omp parallel for
    for (int i = 0; i < this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      int dest_row_length = kv_row_lengths[layer_idx];
      bool is_sliding = dest_row_length == generation_sliding_kv_past_length;
      int src_row_length = context_size;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)this->kvs[kv_idx];
      // key cache or value cache
      int num_column = 128;

      int target_idx = kv_len;
      if (is_sliding && kv_len + _chunk_len > dest_row_length) {
        target_idx = dest_row_length - _chunk_len;
        if(is_key) {
          for(int col = 0; col < num_column; ++col) {
            uint8_t *col_base = dest + col * dest_row_length;
            std::memmove(col_base, col_base + _chunk_len, dest_row_length - _chunk_len);
          }
        } else {
          std::memmove(dest, dest + _chunk_len * num_column, (dest_row_length - _chunk_len) * num_column);
        }
      }

      if (is_key) {
        process_key (output, _chunk_len, num_column, dest, target_idx, dest_row_length, src_row_length);
      } else {
        process_value (output, _chunk_len, num_column, dest, target_idx);
      }
    };

    std::cout << "Prefill KV Cache copy finished!" << std::endl;
  }

  std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
  std::fill_n(generation_sliding_attention_mask,
              generation_sliding_attention_mask_elements, 0);

  generation_attention_mask[generation_attention_mask_elements - 1] =
      std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask
      [generation_sliding_attention_mask_elements - 1] =
          std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < _len && i < generation_full_kv_past_length; i++)
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  int len = (_len <= generation_sliding_kv_past_length)
                ? _len
                : generation_sliding_kv_past_length;
  for (int i = 0; i < len; i++) {
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  }

  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = _len;
  for (idx = prefill_len; idx < generation_full_kv_past_length; idx++) {
    generation_sample[0] = token;

    generation_attention_mask[idx] = std::numeric_limits<uint16_t>::max();
    if (idx < generation_sliding_kv_past_length) {
      generation_sliding_attention_mask[idx] =
          std::numeric_limits<uint16_t>::max();
    }
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

    if (idx > prefill_len) {
#pragma omp parallel for
      for (int i = 0; i < this->kvs.size(); i++) {
        bool is_key = i % 4 > 1;
        int kv_idx = i / 4 * 4 + (i + 2) % 4;
        int layer_idx = i / 4;
        int dest_row_length = kv_row_lengths[layer_idx];
        bool is_sliding = dest_row_length == generation_sliding_kv_past_length;
        int src_row_length = 1;

        auto output = std::get<uint8_t *>(outputs[i]);
        auto dest = (uint8_t *)this->kvs[kv_idx];
        // key cache or value cache
        int num_row = 1;
        int num_column = 128;

        int target_idx = idx;
        if (is_sliding && (idx + 1) > dest_row_length) {
          target_idx = dest_row_length - 1;
          if (is_key) {
            for (int col = 0; col < num_column; ++col) {
              uint8_t *col_base = dest + col * dest_row_length;
              std::memmove (col_base, col_base + 1, dest_row_length - 1);
            }
          } else {
            std::memmove (dest, dest + num_column, (dest_row_length - 1) * num_column);
          }
        }

        if (is_key) {
          process_key (output, 1, num_column, dest, target_idx, dest_row_length, 1);
        } else {
          process_value (output, 1, num_column, dest, target_idx);
        }
      };
    }

    outputs = generation_model->inference(1, generation_inputs);
    auto *generation_logits = std::get<uint16_t *>(outputs.back());
    log_eos_debug(generation_logits, vocab_size, eos_token, logit_scale,
                  logit_offset, temperature, idx - prefill_len);

    if (do_sample) {
      token = sample_gauss3_6_corrected(
          generation_logits, vocab_size, _input.data(), _input.size(),
          logit_scale, logit_offset, repetition_penalty, temperature, top_p,
          top_k);
    } else {
      token = select_greedy_token(generation_logits, vocab_size);
    }

    output.push_back(token);
    if (token == eos_token) {
      std::cout << "Finished generating, break..." << std::endl;
      break;
    } else {
      std::string decoded= tokenizer->Decode({token});
      last_output_ += decoded;
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
      _input.push_back(token);
    }

    // Cooperative cancellation: a streamer may have asked us to stop
    // via its put() return value, or requestStop() was called from
    // another thread. We check once per generated token so worst-case
    // latency is a single decode step.
    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
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
            << ", token per second: " << (idx - _len) / raw_exec_seconds.count()
            << ", token generation time average: "
            << raw_exec_seconds.count() / (idx - _len) << std::endl;
  }
}
