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

#include "streamer.h"
#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <stdexcept>

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

  LOGD("----------------------- initialize() %s, %s", prefill_graph.c_str(),
       generation_graph.c_str());
  // Get references to graph_info and model_inputs
  auto &prefill_graph_info = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;
  LOGD("----------------------- initialize() prefill_inputs : %lu",
       prefill_inputs.size());

  prefill_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      prefill_graph_info.raw_inputs, "attention_mask");
  prefill_sliding_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      prefill_graph_info.raw_inputs, "sliding_attention_mask");
  generation_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      generation_graph_info.raw_inputs, "attention_mask");
  generation_sliding_attention_mask_elements =
    GraphParser::get_named_tensor_elements_or_throw(
      generation_graph_info.raw_inputs, "sliding_attention_mask");
  generation_full_kv_past_length = generation_attention_mask_elements - 1;
  generation_sliding_kv_past_length =
    generation_sliding_attention_mask_elements - 1;
  rope_cache_seq_len =
    std::max(max_seq_len, generation_attention_mask_elements);

  // Find input indices by name
  int prefill_input_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "inputs_embeds");
  int generation_input_idx = GraphParser::find_tensor_index(
    generation_graph_info.raw_inputs, "inputs_embeds");
  LOGD("----------------------- prefill_inputs_idx : %d generation_input_idx "
       ":%d ",
       prefill_input_idx, generation_input_idx);
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
  LOGD("----------------------- initialize() 1");

  // SWA Position IDs
  int prefill_swa_pos_cos_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "swa_position_ids_cos");
  int prefill_swa_pos_sin_idx = GraphParser::find_tensor_index(
    prefill_graph_info.raw_inputs, "swa_position_ids_sin");
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
      const auto &info = prefill_graph_info.raw_inputs[idx];
      if (info.name.find("_lora_") != std::string::npos) {
        int size = GraphParser::get_tensor_size(info);
        memcpy(std::get<uint16_t *>(prefill_inputs[idx]), data_ptr, size);
        data_ptr += size;
      }
    }
    LOGD("----------------------- initialize() 6");
    // Copy to generation lora inputs (in model input order)
    for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
      const auto &info = generation_graph_info.raw_inputs[idx];
      if (info.name.find("_lora_") != std::string::npos) {
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

  // Initialize generation KV cache as the canonical history. Prefill keeps its
  // own input buffers because gauss3.6 prefill and generation KV shapes differ.
  kv_cache_.clear();
  LOGD("----------------------- initialize() 8");

  for (int layer = 0; layer < num_hidden_layers; layer++) {
    const std::string key_h0_name =
      "past_key_" + std::to_string(layer) + "_h0_in";
    const auto &generation_key_h0_info = GraphParser::get_tensor_info_or_throw(
      generation_graph_info.raw_inputs, key_h0_name);
    kv_cache_.addLayerRowLength(generation_key_h0_info.dimensions.back());

    const std::vector<std::string> kv_names = {
      key_h0_name,
      "past_key_" + std::to_string(layer) + "_h1_in",
      "past_value_" + std::to_string(layer) + "_h0_in",
      "past_value_" + std::to_string(layer) + "_h1_in",
    };

    for (const auto &name : kv_names) {
      int generation_input_index =
        GraphParser::find_tensor_index(generation_graph_info.raw_inputs, name);
      int prefill_input_index =
        find_tensor_index_or_minus_one(prefill_graph_info.raw_inputs, name);
      const auto &generation_info =
        generation_graph_info.raw_inputs[generation_input_index];

      const bool is_key = qnn_starts_with(name, "past_key_");
      int size = GraphParser::get_tensor_size(generation_info);
      if (generation_info.data_type != "QNN_DATATYPE_UFIXED_POINT_8") {
        throw std::runtime_error("Unexpected generation KV dtype for " + name);
      }

      auto *current_kv =
        std::get<uint8_t *>(generation_inputs[generation_input_index]);
      int kv_input_index = kv_cache_.addGenerationCache(
        name, current_kv, size,
        get_kv_row_length(generation_info, is_key, name), is_key);

      if (prefill_input_index >= 0) {
        const auto &prefill_info =
          prefill_graph_info.raw_inputs[prefill_input_index];

        kv_cache_.addPrefillCache(
          std::get<uint8_t *>(prefill_inputs[prefill_input_index]),
          GraphParser::get_tensor_size(prefill_info),
          get_kv_row_length(prefill_info, is_key, name), kv_input_index,
          is_key);
      }
    }
  }

  kv_cache_.setPrefillOutputBindings(
    build_kv_output_bindings(prefill_graph_info.raw_outputs,
                             kv_cache_.generationIndexByName(), prefill_graph));
  kv_cache_.setGenerationOutputBindings(build_kv_output_bindings(
    generation_graph_info.raw_outputs, kv_cache_.generationIndexByName(),
    generation_graph));

  LOGD("KV cache mapping: generation_inputs=%zu prefill_inputs=%zu "
       "prefill_outputs=%zu generation_outputs=%zu",
       kv_cache_.generationCacheCount(), kv_cache_.prefillCacheCount(),
       kv_cache_.prefillOutputBindingCount(),
       kv_cache_.generationOutputBindingCount());

  initialize_kv_cache();
  LOGD("----------------------- initialize() done");
}

void causallm::Gauss3_6_QNN::initialize_kv_cache() { kv_cache_.reset(); }

void causallm::Gauss3_6_QNN::reset_prefill_kv_cache_inputs() {
  kv_cache_.resetPrefillInputs();
}

void causallm::Gauss3_6_QNN::sync_generation_kv_cache_to_prefill() {
  kv_cache_.syncGenerationToPrefill();
}

void causallm::Gauss3_6_QNN::resetKvCache() { kv_cache_.reset(); }

void causallm::Gauss3_6_QNN::saveKvCache(const std::string &cache_path) const {
  kv_cache_.save(cache_path, architectures);
}

void causallm::Gauss3_6_QNN::loadKvCache(const std::string &cache_path) {
  kv_cache_.load(cache_path, architectures, generation_full_kv_past_length);
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
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  resetKvCache();

  stop_requested_.store(false, std::memory_order_release);

  const std::string model_prompt = promptToUtf8(prompt);
  auto input = tokenizer->Encode(model_prompt);

  if (input.size() <= 1) {
    std::cout << "[Error] Input is empty or invalid" << std::endl;
    return;
  }
  unsigned int input_len = input.size() - 1;

  if (kv_cache_.length() + static_cast<int>(input_len) >=
      generation_full_kv_past_length) {
    throw std::runtime_error(
      "Input prompt leaves no room for generation: kv_len=" +
      std::to_string(kv_cache_.length()) + ", input_len=" +
      std::to_string(input_len) + ", generation_full_kv_past_length=" +
      std::to_string(generation_full_kv_past_length));
  }

  auto n_chunks = (input_len % context_size != 0)
                    ? ((input_len / context_size) + 1)
                    : (input_len / context_size);
  auto token = input.back();

  std::cout << "len: " << input_len << ", n_chunks: " << n_chunks << std::endl;
  LOGD("prompt token length=%u, n_chunks=%u, full_kv_past=%d, "
       "sliding_kv_past=%d, rope_cache_seq_len=%d, model_prompt_bytes=%zu",
       input_len, n_chunks, generation_full_kv_past_length,
       generation_sliding_kv_past_length, rope_cache_seq_len,
       model_prompt.size());

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

  auto append_generation_token_to_kv_cache = [&](int token_to_append) {
    const int current_kv_len = kv_cache_.length();
    if (current_kv_len >= generation_full_kv_past_length) {
      LOGD("skip appending terminal token to KV: kv_len=%d, full_kv_past=%d",
           current_kv_len, generation_full_kv_past_length);
      return;
    }

    fill_generation_inputs(
      generation_sample, token_to_append, generation_attention_mask,
      generation_attention_mask_elements, generation_sliding_attention_mask,
      generation_sliding_attention_mask_elements,
      generation_full_kv_past_length, generation_sliding_kv_past_length,
      generation_position_ids_cos, generation_position_ids_sin,
      position_ids_cos, position_ids_sin, pos_dim,
      generation_swa_position_ids_cos, generation_swa_position_ids_sin,
      swa_position_ids_cos, swa_position_ids_sin, pos_dim, current_kv_len,
      rope_cache_seq_len);
    auto terminal_outputs = generation_model->inference(1, generation_inputs);
    kv_cache_.appendGenerationOutputs(terminal_outputs, current_kv_len, 1, 1,
                                      generation_graph);
    kv_cache_.advance(1);
  };

  auto start_prefill = std::chrono::system_clock::now();

  for (int c = 0; c < n_chunks; c++) {
    const int chunk_offset = c * context_size;
    int chunk_len = ((c + 1) * context_size < input_len)
                      ? context_size
                      : (input_len - chunk_offset);
    const int current_kv_len = kv_cache_.length();
    LOGD("kv_len : %d, chunk_len : %d", current_kv_len, chunk_len);
    sync_generation_kv_cache_to_prefill();

    for (int i = 0; i < context_size; i++)
      input_sample[i] =
        (i < chunk_len) ? input[chunk_offset + i] : padding_token;

    fill_attention_mask_with_length(context_size, max_seq_len, chunk_len,
                                    attention_mask);
    fill_attention_mask_with_prev_length(
      context_size, max_seq_len,
      std::min(current_kv_len, max_seq_len - context_size), attention_mask);

    if (current_kv_len >= generation_sliding_kv_past_length) {
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
      fill_attention_mask_with_prev_length(
        context_size, sliding_window, current_kv_len, sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    if (current_kv_len + chunk_len > rope_cache_seq_len) {
      throw std::runtime_error("Prefill position is out of rope cache");
    }

    auto pos_ids_offset = current_kv_len * pos_dim;
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
    kv_cache_.appendPrefillOutputs(outputs, current_kv_len, chunk_len,
                                   context_size, prefill_graph);

    kv_cache_.advance(chunk_len);
  }

  auto end_prefill = std::chrono::system_clock::now();
  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = kv_cache_.length();
  for (idx = prefill_len; idx < generation_full_kv_past_length; idx++) {
    fill_generation_inputs(
      generation_sample, token, generation_attention_mask,
      generation_attention_mask_elements, generation_sliding_attention_mask,
      generation_sliding_attention_mask_elements,
      generation_full_kv_past_length, generation_sliding_kv_past_length,
      generation_position_ids_cos, generation_position_ids_sin,
      position_ids_cos, position_ids_sin, pos_dim,
      generation_swa_position_ids_cos, generation_swa_position_ids_sin,
      swa_position_ids_cos, swa_position_ids_sin, pos_dim, idx,
      rope_cache_seq_len);

    outputs = generation_model->inference(1, generation_inputs);
    kv_cache_.appendGenerationOutputs(outputs, idx, 1, 1, generation_graph);
    kv_cache_.advance(1);
    token = sample(std::get<uint16_t *>(outputs.back()), vocab_size,
                   input.data(), input.size(), logit_scale, logit_offset,
                   repetition_penalty, temperature, top_p, top_k);

    output.push_back(token);
    if (token == eos_token) {
      append_generation_token_to_kv_cache(token);
      break;
    } else {
      std::string decoded = tokenizer->Decode({token});
      LOGD("%d : %s", token, decoded.c_str());
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

  auto end = std::chrono::system_clock::now();
  this->raw_exec_seconds = end - start;

  auto prefill_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_prefill - start_prefill)
                      .count();
  auto gen_ms =
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  unsigned int generated_tokens =
    (idx > prefill_len) ? static_cast<unsigned int>(idx - prefill_len) : 0U;

  performance_metrics.prefill_tokens = static_cast<unsigned int>(input_len);
  performance_metrics.prefill_duration_ms = static_cast<double>(prefill_ms);
  performance_metrics.generation_tokens = generated_tokens;
  performance_metrics.generation_duration_ms = static_cast<double>(gen_ms);
  performance_metrics.total_duration_ms =
    static_cast<double>(prefill_ms + gen_ms);
  performance_metrics.peak_memory_kb = getPeakMemoryKb();

  has_run_ = true;

  if (log_output) {
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Generation exec_time : " << this->raw_exec_seconds.count()
              << ", token per second: "
              << generated_tokens / this->raw_exec_seconds.count()
              << ", token generation time average: "
              << this->raw_exec_seconds.count() /
                   std::max(1, static_cast<int>(generated_tokens))
              << std::endl;
  }
}
