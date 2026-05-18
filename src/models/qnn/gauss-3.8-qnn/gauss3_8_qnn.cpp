// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_8_qnn.cpp
 * @brief  QNN model implementation with self-registration
 */

#include "gauss3_8_qnn.h"
#include "android_memory_allocator.h"
#include "generate_qnn_utils.h"

#include <streamer.h>
#include <llm_util.hpp>
#include <model.h>
#include <xgrammar/xgrammar_wrapper.h>

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
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using namespace causallm;

__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "Gauss_3_8_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                          causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gauss3_8_QNN>(cfg, generation_cfg,
                                                        nntr_cfg);
      });
}

causallm::Gauss3_8_QNN::~Gauss3_8_QNN() {
  if (embedding_mmap_ptr != nullptr) {
    ::munmap(embedding_mmap_ptr, embedding_mmap_size);
    embedding_mmap_ptr = nullptr;
    embedding_mmap_size = 0;
    embedding_bytes_per_token = 0;
  }
}

void causallm::Gauss3_8_QNN::initialize() {
  Quick_Dot_AI_QNN::initialize();
  LOGD("Quick_Dot_AI_QNN::initialize() done");

  if (graphs_to_use.size() < 2) {
    throw std::runtime_error("Gauss3_8_QNN requires prefill and generation graphs");
  }

  const std::string prefill_graph = graphs_to_use[0];
  const std::string generation_graph = graphs_to_use[1];

  auto &prefill_graph_info = models[prefill_graph].graph_info;
  auto &generation_graph_info = models[generation_graph].graph_info;
  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;

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

  const int prefill_input_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "inputs_embeds");
  const int generation_input_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "inputs_embeds");

  if (uses_embedding) {
    input_sample = std::get<float *>(prefill_inputs[prefill_input_idx]);
    generation_sample =
        std::get<float *>(generation_inputs[generation_input_idx]);
    input_sample_u16 = nullptr;
    generation_sample_u16 = nullptr;
  } else {
    input_sample_u16 =
        std::get<uint16_t *>(prefill_inputs[prefill_input_idx]);
    generation_sample_u16 =
        std::get<uint16_t *>(generation_inputs[generation_input_idx]);
    input_sample = nullptr;
    generation_sample = nullptr;
  }

  const int prefill_attn_mask_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "attention_mask");
  const int prefill_sliding_attn_mask_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "sliding_attention_mask");
  const int generation_attn_mask_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "attention_mask");
  const int generation_sliding_attn_mask_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "sliding_attention_mask");

  attention_mask = std::get<uint16_t *>(prefill_inputs[prefill_attn_mask_idx]);
  sliding_attention_mask =
      std::get<uint16_t *>(prefill_inputs[prefill_sliding_attn_mask_idx]);
  generation_attention_mask =
      std::get<uint16_t *>(generation_inputs[generation_attn_mask_idx]);
  generation_sliding_attention_mask = std::get<uint16_t *>(
      generation_inputs[generation_sliding_attn_mask_idx]);

  const int prefill_pos_cos_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "position_ids_cos");
  const int prefill_pos_sin_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "position_ids_sin");
  const int generation_pos_cos_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "position_ids_cos");
  const int generation_pos_sin_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "position_ids_sin");

  prefill_position_ids_cos =
      std::get<uint16_t *>(prefill_inputs[prefill_pos_cos_idx]);
  prefill_position_ids_sin =
      std::get<uint16_t *>(prefill_inputs[prefill_pos_sin_idx]);
  generation_position_ids_cos =
      std::get<uint16_t *>(generation_inputs[generation_pos_cos_idx]);
  generation_position_ids_sin =
      std::get<uint16_t *>(generation_inputs[generation_pos_sin_idx]);

  const int prefill_swa_pos_cos_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "swa_position_ids_cos");
  const int prefill_swa_pos_sin_idx =
      GraphParser::find_tensor_index(prefill_graph_info.raw_inputs,
                                     "swa_position_ids_sin");
  const int generation_swa_pos_cos_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "swa_position_ids_cos");
  const int generation_swa_pos_sin_idx =
      GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                     "swa_position_ids_sin");

  prefill_swa_position_ids_cos =
      std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_cos_idx]);
  prefill_swa_position_ids_sin =
      std::get<uint16_t *>(prefill_inputs[prefill_swa_pos_sin_idx]);
  generation_swa_position_ids_cos =
      std::get<uint16_t *>(generation_inputs[generation_swa_pos_cos_idx]);
  generation_swa_position_ids_sin =
      std::get<uint16_t *>(generation_inputs[generation_swa_pos_sin_idx]);

  auto cos_sin_tuple = get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);
  allocated_ptrs_.insert(position_ids_cos);
  allocated_ptrs_.insert(position_ids_sin);

  auto swa_cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, pos_dim, local_rope_theta);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  allocated_ptrs_.insert(swa_position_ids_cos);
  allocated_ptrs_.insert(swa_position_ids_sin);

  auto fill_lora_inputs =
      [](const TensorInfoList &raw_inputs,
         std::vector<ml::train::TensorDim::IO_TensorType> &model_inputs) {
        for (size_t idx = 0; idx < raw_inputs.size(); idx++) {
          const auto &[name, info] = raw_inputs[idx];
          if (name.find("_lora_") == std::string::npos) {
            continue;
          }
          const int size = GraphParser::get_tensor_size(info);
          auto *lora_ptr = std::get<uint16_t *>(model_inputs[idx]);
          std::fill_n(lora_ptr, size / sizeof(uint16_t), 32768);
        }
      };

  if (lora_path.empty()) {
    fill_lora_inputs(prefill_graph_info.raw_inputs, prefill_inputs);
    fill_lora_inputs(generation_graph_info.raw_inputs, generation_inputs);
  } else {
    int fd = open(lora_path.c_str(), O_RDONLY);
    if (fd < 0) {
      throw std::runtime_error("Failed to open lora_path: " + lora_path);
    }

    struct stat st {};
    if (fstat(fd, &st) < 0) {
      close(fd);
      throw std::runtime_error("Failed to stat lora_path: " + lora_path);
    }
    const size_t file_size = st.st_size;

    void *mapped = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped == MAP_FAILED) {
      close(fd);
      throw std::runtime_error("Failed to mmap lora_path: " + lora_path);
    }

    uint8_t *data_ptr = static_cast<uint8_t *>(mapped);
    uint8_t *data_end = data_ptr + file_size;
    std::unordered_set<std::string> copied_names;

    std::unordered_map<std::string, std::pair<uint16_t *, int>>
        generation_lora_by_name;
    for (size_t idx = 0; idx < generation_graph_info.raw_inputs.size(); idx++) {
      const auto &[name, info] = generation_graph_info.raw_inputs[idx];
      if (name.find("_lora_") == std::string::npos) {
        continue;
      }
      generation_lora_by_name[name] = {
          std::get<uint16_t *>(generation_inputs[idx]),
          GraphParser::get_tensor_size(info)};
    }

    auto copy_one_lora = [&](const std::string &name, uint16_t *prefill_ptr,
                             int size) {
      if (data_ptr + size > data_end) {
        munmap(mapped, file_size);
        close(fd);
        throw std::runtime_error("LoRA file is shorter than graph metadata");
      }
      std::memcpy(prefill_ptr, data_ptr, size);

      auto gen_it = generation_lora_by_name.find(name);
      if (gen_it != generation_lora_by_name.end()) {
        if (gen_it->second.second != size) {
          munmap(mapped, file_size);
          close(fd);
          throw std::runtime_error("LoRA size mismatch for " + name);
        }
        std::memcpy(gen_it->second.first, data_ptr, size);
      }

      data_ptr += size;
      copied_names.insert(name);
    };

    for (size_t idx = 0; idx < prefill_graph_info.raw_inputs.size(); idx++) {
      const auto &[name, info] = prefill_graph_info.raw_inputs[idx];
      if (name.find("_lora_") == std::string::npos) {
        continue;
      }
      copy_one_lora(name, std::get<uint16_t *>(prefill_inputs[idx]),
                    GraphParser::get_tensor_size(info));
    }

    for (const auto &[name, binding] : generation_lora_by_name) {
      if (copied_names.find(name) != copied_names.end()) {
        continue;
      }
      if (data_ptr + binding.second > data_end) {
        munmap(mapped, file_size);
        close(fd);
        throw std::runtime_error("LoRA file is shorter than graph metadata");
      }
      std::memcpy(binding.first, data_ptr, binding.second);
      data_ptr += binding.second;
    }

    munmap(mapped, file_size);
    close(fd);

    std::cout << "LoRA weights loaded from: " << lora_path << std::endl;
  }

  if (!uses_embedding) {
    if (embedding_mmap_ptr != nullptr) {
      ::munmap(embedding_mmap_ptr, embedding_mmap_size);
      embedding_mmap_ptr = nullptr;
      embedding_mmap_size = 0;
      embedding_bytes_per_token = 0;
    }

    if (!embedding_file_name.empty()) {
      int fd = ::open(embedding_file_name.c_str(), O_RDONLY);
      if (fd < 0) {
        LOGE("Gauss3_8_QNN: open embedding file failed: %s",
             embedding_file_name.c_str());
      } else {
        struct stat st {};
        if (::fstat(fd, &st) == 0) {
          embedding_mmap_size = static_cast<size_t>(st.st_size);
          embedding_mmap_ptr =
              ::mmap(nullptr, embedding_mmap_size, PROT_READ, MAP_PRIVATE, fd,
                     0);
          if (embedding_mmap_ptr == MAP_FAILED) {
            LOGE("Gauss3_8_QNN: mmap embedding file failed");
            embedding_mmap_ptr = nullptr;
            embedding_mmap_size = 0;
          } else {
            (void)::posix_madvise(embedding_mmap_ptr, embedding_mmap_size,
                                  POSIX_MADV_RANDOM);
            embedding_bytes_per_token = hidden_size * sizeof(uint16_t);
            LOGD("Gauss3_8_QNN: embedding table mmap'd (%zu bytes, "
                 "per-token=%zu, vocab=%zu)",
                 embedding_mmap_size, embedding_bytes_per_token,
                 embedding_mmap_size / embedding_bytes_per_token);
          }
        }
        ::close(fd);
      }
    }
  }

  kv_cache_.clear();

  for (int layer = 0; layer < num_hidden_layers; layer++) {
    const std::string key_h0_name =
        "past_key_" + std::to_string(layer) + "_h0_in";
    const auto &generation_key_h0_info =
        GraphParser::get_tensor_info_or_throw(generation_graph_info.raw_inputs,
                                              key_h0_name);
    kv_cache_.addLayerRowLength(generation_key_h0_info.dimensions.back());

    const std::vector<std::string> kv_names = {
        key_h0_name,
        "past_key_" + std::to_string(layer) + "_h1_in",
        "past_value_" + std::to_string(layer) + "_h0_in",
        "past_value_" + std::to_string(layer) + "_h1_in",
    };

    for (const auto &name : kv_names) {
      const int generation_input_index =
          GraphParser::find_tensor_index(generation_graph_info.raw_inputs,
                                         name);
      const int prefill_input_index =
          find_tensor_index_or_minus_one(prefill_graph_info.raw_inputs, name);
      const auto &generation_info =
          generation_graph_info.raw_inputs[generation_input_index].second;
      const int size = GraphParser::get_tensor_size(generation_info);
      if (generation_info.data_type != "QNN_DATATYPE_UFIXED_POINT_8") {
        throw std::runtime_error("Unexpected generation KV dtype for " + name);
      }

      auto *current_kv =
          std::get<uint8_t *>(generation_inputs[generation_input_index]);
      std::memset(current_kv, 128, size);

      const bool is_key = qnn_starts_with(name, "past_key_");
      const int kv_input_index = kv_cache_.addGenerationCache(
          name, current_kv, size,
          get_kv_row_length(generation_info, is_key, name), is_key);

      if (prefill_input_index >= 0) {
        const auto &prefill_info =
            prefill_graph_info.raw_inputs[prefill_input_index].second;
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
                               kv_cache_.generationIndexByName(),
                               prefill_graph));
  kv_cache_.setGenerationOutputBindings(
      build_kv_output_bindings(generation_graph_info.raw_outputs,
                               kv_cache_.generationIndexByName(),
                               generation_graph));

  generation_logits_output_index = find_tensor_index_or_minus_one(
      generation_graph_info.raw_outputs, "logits");
  if (generation_logits_output_index < 0) {
    generation_logits_output_index =
        static_cast<int>(generation_graph_info.raw_outputs.size()) - 1;
  }

  LOGD("KV cache mapping: generation_inputs=%zu prefill_inputs=%zu "
       "prefill_outputs=%zu generation_outputs=%zu logits_output=%d",
       kv_cache_.generationCacheCount(), kv_cache_.prefillCacheCount(),
       kv_cache_.prefillOutputBindingCount(),
       kv_cache_.generationOutputBindingCount(), generation_logits_output_index);

  initialize_kv_cache();
}

void causallm::Gauss3_8_QNN::initialize_kv_cache() {
  kv_cache_.reset();
}

void causallm::Gauss3_8_QNN::resetKvCache() { kv_cache_.reset(); }

void causallm::Gauss3_8_QNN::saveKvCache(
    const std::string &cache_path) const {
  kv_cache_.save(cache_path, architectures);
}

void causallm::Gauss3_8_QNN::loadKvCache(const std::string &cache_path) {
  kv_cache_.load(cache_path, architectures, generation_full_kv_past_length);
}

void causallm::Gauss3_8_QNN::setupParameters(json &cfg,
                                             json &generation_cfg,
                                             json &nntr_cfg) {
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  num_hidden_layers = cfg["num_hidden_layers"].get<int>();
  max_window_layers = cfg["max_window_layers"].get<int>();
  hidden_size = cfg["hidden_size"].get<int>();
  sequence_length = cfg["sequence_length"].get<int>();
  max_seq_len = cfg["max_seq_len"].get<int>();
  sliding_window = cfg["sliding_window"].get<int>();
  local_rope_theta = cfg["local_rope_theta"].get<float>();
  rope_theta = cfg["rope_theta"].get<float>();
  context_size = cfg["context_size"].get<int>();
  pos_dim = cfg["pos_dim"].get<int>();
  head_dim = cfg["head_dim"].get<int>();

  padding_token = generation_cfg["padding_token"].get<int>();
  eos_token = generation_cfg["eos_token_id"].get<int>();
  temperature = generation_cfg["temperature"].get<float>();
  top_k = generation_cfg["top_k"].get<int>();
  top_p = generation_cfg["top_p"].get<float>();
  repetition_penalty = generation_cfg["repetition_penalty"].get<float>();
  logit_scale = generation_cfg["logit_scale"].get<float>();
  logit_offset = generation_cfg["logit_offset"].get<int>();

  lora_path = nntr_cfg.value("lora_path", "");
}

void causallm::Gauss3_8_QNN::run(const WSTR prompt, bool do_sample,
                                 const WSTR system_prompt,
                                 const WSTR tail_prompt, bool log_output) {
  (void)do_sample;
  (void)system_prompt;
  (void)tail_prompt;

  stop_requested_.store(false, std::memory_order_release);

  const std::string model_prompt = promptToUtf8(prompt);
  auto input = tokenizer->Encode(model_prompt);

  if (input.size() <= 1) {
    std::cout << "[Error] Input is empty or invalid" << std::endl;
    return;
  }

  const unsigned int input_len = input.size() - 1;
  if (kv_cache_.length() + static_cast<int>(input_len) >=
      generation_full_kv_past_length) {
    throw std::runtime_error(
        "Input prompt leaves no room for generation: kv_len=" +
        std::to_string(kv_cache_.length()) +
        ", input_len=" + std::to_string(input_len) +
        ", generation_full_kv_past_length=" +
        std::to_string(generation_full_kv_past_length));
  }

  const unsigned int n_chunks =
      (input_len % context_size != 0) ? ((input_len / context_size) + 1)
                                      : (input_len / context_size);
  int token = input.back();

  std::cout << "len: " << input_len << ", n_chunks: " << n_chunks
            << std::endl;
  LOGD("prompt token length=%u, n_chunks=%u, full_kv_past=%d, "
       "sliding_kv_past=%d, rope_cache_seq_len=%d, model_prompt_bytes=%zu",
       input_len, n_chunks, generation_full_kv_past_length,
       generation_sliding_kv_past_length, rope_cache_seq_len,
       model_prompt.size());

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  const std::string prefill_graph = graphs_to_use[0];
  const std::string generation_graph = graphs_to_use[1];

  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;
  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  auto copy_token_embedding = [&](uint16_t *dest, int token_id) {
    const void *embedding = lookupEmbedding(token_id);
    if (embedding == nullptr) {
      throw std::runtime_error("lookupEmbedding(" + std::to_string(token_id) +
                               ") returned null");
    }
    std::memcpy(dest, embedding, embedding_bytes_per_token);
  };

  auto fill_prefill_inputs = [&](int chunk_offset, int chunk_len) {
    if (uses_embedding) {
      for (int i = 0; i < context_size; i++) {
        input_sample[i] =
            (i < chunk_len) ? input[chunk_offset + i] : padding_token;
      }
      return;
    }

    for (int i = 0; i < context_size; i++) {
      const int token_id =
          (i < chunk_len) ? input[chunk_offset + i] : padding_token;
      copy_token_embedding(input_sample_u16 + i * hidden_size, token_id);
    }
  };

  auto fill_generation_inputs = [&](int current_token, int position) {
    if (position < 0 || position >= rope_cache_seq_len) {
      throw std::runtime_error("Generation position is out of rope cache");
    }

    if (uses_embedding) {
      generation_sample[0] = current_token;
    } else {
      copy_token_embedding(generation_sample_u16, current_token);
    }

    std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
    std::fill_n(generation_sliding_attention_mask,
                generation_sliding_attention_mask_elements, 0);

    generation_attention_mask[generation_attention_mask_elements - 1] =
        std::numeric_limits<uint16_t>::max();
    generation_sliding_attention_mask
        [generation_sliding_attention_mask_elements - 1] =
            std::numeric_limits<uint16_t>::max();

    for (int i = 0; i < position && i < generation_full_kv_past_length; i++) {
      generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
    }
    for (int i = 0; i < position && i < generation_sliding_kv_past_length;
         i++) {
      generation_sliding_attention_mask[i] =
          std::numeric_limits<uint16_t>::max();
    }

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

  auto append_generation_token_to_kv_cache = [&](int token_to_append) {
    if (kv_cache_.length() >= generation_full_kv_past_length) {
      LOGD("skip appending terminal token to KV: kv_len=%d, full_kv_past=%d",
           kv_cache_.length(), generation_full_kv_past_length);
      return;
    }

    fill_generation_inputs(token_to_append, kv_cache_.length());
    auto terminal_outputs = generation_model->inference(1, generation_inputs);
    kv_cache_.appendGenerationOutputs(terminal_outputs, kv_cache_.length(), 1,
                                      1, generation_graph);
    kv_cache_.advance(1);
  };

  auto prefill_start = std::chrono::high_resolution_clock::now();

  for (unsigned int c = 0; c < n_chunks; c++) {
    const int chunk_offset = c * context_size;
    const int chunk_len =
        ((c + 1) * context_size < input_len)
            ? context_size
            : (static_cast<int>(input_len) - chunk_offset);

    LOGD("kv_len: %d, chunk_len: %d", kv_cache_.length(), chunk_len);
    kv_cache_.syncGenerationToPrefill();
    fill_prefill_inputs(chunk_offset, chunk_len);

    fill_attention_mask_with_length(context_size, max_seq_len, chunk_len,
                                    attention_mask);
    fill_attention_mask_with_prev_length(
        context_size, max_seq_len,
        std::min(kv_cache_.length(), max_seq_len - context_size),
        attention_mask);

    if (kv_cache_.length() >= generation_sliding_kv_past_length) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window,
                  std::numeric_limits<uint16_t>::min());
      for (int i = 0; i < chunk_len; i++) {
        for (int j = i + 1; j < i + generation_sliding_attention_mask_elements;
             j++) {
          sliding_attention_mask[i * sliding_window + j] =
              std::numeric_limits<uint16_t>::max();
        }
      }
    } else {
      fill_attention_mask_with_length(context_size, sliding_window, chunk_len,
                                      sliding_attention_mask);
      fill_attention_mask_with_prev_length(context_size, sliding_window,
                                           kv_cache_.length(),
                                           sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    if (kv_cache_.length() + chunk_len > rope_cache_seq_len) {
      throw std::runtime_error("Prefill position is out of rope cache");
    }

    const int pos_ids_offset = kv_cache_.length() * pos_dim;
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

    outputs = prefill_model->inference(1, prefill_inputs);
    kv_cache_.appendPrefillOutputs(outputs, kv_cache_.length(), chunk_len,
                                   context_size, prefill_graph);
    kv_cache_.advance(chunk_len);
  }
  auto prefill_end = std::chrono::high_resolution_clock::now();

  auto start = std::chrono::system_clock::now();
  int idx = kv_cache_.length();
  const int prefill_len = kv_cache_.length();
  for (; idx < generation_full_kv_past_length; idx++) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
    }

    fill_generation_inputs(token, idx);

    outputs = generation_model->inference(1, generation_inputs);
    kv_cache_.appendGenerationOutputs(outputs, idx, 1, 1, generation_graph);
    kv_cache_.advance(1);

    token = sample(std::get<uint16_t *>(outputs[generation_logits_output_index]),
                   vocab_size, input.data(), input.size(), logit_scale,
                   logit_offset, repetition_penalty, temperature, top_p, top_k);

    output.push_back(token);
    if (token == eos_token || token == padding_token) {
      append_generation_token_to_kv_cache(token);
      break;
    }

    std::string decoded = tokenizer->Decode({token});
    LOGD("%d : %s (idx: %d)", token, decoded.c_str(), idx);
    if (streamer_) {
      if (streamer_put(streamer_, decoded.c_str()) != 0) {
        stop_requested_.store(true, std::memory_order_release);
        break;
      }
    } else if (log_output) {
      std::cout << decoded << std::flush;
    }
    input.push_back(token);
  }

  if (streamer_) {
    streamer_end(streamer_);
  }

  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;

  performance_metrics.prefill_tokens = input_len;
  performance_metrics.prefill_duration_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  performance_metrics.generation_tokens = (idx > prefill_len) ? (unsigned int)(idx - prefill_len) : 0U;
  performance_metrics.generation_duration_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  performance_metrics.total_duration_ms =
      performance_metrics.prefill_duration_ms + performance_metrics.generation_duration_ms;
  performance_metrics.peak_memory_kb = 0; // TODO: implement memory tracking

  has_run_ = true;
  if (log_output) {
    const int generated = std::max(1, idx - prefill_len);
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Generation exec_time : " << raw_exec_seconds.count()
              << ", token per second: "
              << generated / raw_exec_seconds.count()
              << ", token generation time average: "
              << raw_exec_seconds.count() / generated << std::endl;
  }
}

const void *causallm::Gauss3_8_QNN::lookupEmbedding(int token_id) const {
  if (embedding_mmap_ptr == nullptr || embedding_bytes_per_token == 0) {
    return nullptr;
  }
  if (token_id < 0) {
    return nullptr;
  }
  const size_t offset =
      static_cast<size_t>(token_id) * embedding_bytes_per_token;
  if (offset + embedding_bytes_per_token > embedding_mmap_size) {
    return nullptr;
  }
  return static_cast<const uint8_t *>(embedding_mmap_ptr) + offset;
}

std::pair<float, int> causallm::Gauss3_8_QNN::get_embedding_info() {
  return {0.0000029912209811300274f, -34952};
}

void causallm::Gauss3_8_QNN::run_with_embeddings(
    const void *prefill_embeds, size_t n_tokens, std::vector<int> seed_tokens,
    bool do_sample, bool log_output) {
  (void)do_sample;

  stop_requested_.store(false, std::memory_order_release);

  if (input_sample_u16 == nullptr || generation_sample_u16 == nullptr) {
    LOGE("run_with_embeddings: u16 input/generation sample not initialized. "
         "Was uses_embedding=false set?");
    return;
  }
  if (embedding_mmap_ptr == nullptr) {
    LOGE("run_with_embeddings: embedding table not loaded");
    return;
  }
  if (prefill_embeds == nullptr || n_tokens == 0) {
    LOGE("run_with_embeddings: empty prefill_embeds");
    return;
  }

  const unsigned int input_len = static_cast<unsigned int>(n_tokens) - 1;
  if (kv_cache_.length() + static_cast<int>(input_len) >=
      generation_full_kv_past_length) {
    throw std::runtime_error(
        "Input embeddings leave no room for generation: kv_len=" +
        std::to_string(kv_cache_.length()) +
        ", input_len=" + std::to_string(input_len) +
        ", generation_full_kv_past_length=" +
        std::to_string(generation_full_kv_past_length));
  }

  const size_t bytes_per_token = embedding_bytes_per_token;
  const unsigned int n_chunks =
      (input_len % context_size != 0) ? ((input_len / context_size) + 1)
                                      : (input_len / context_size);

  std::cout << "len: " << input_len << ", n_chunks: " << n_chunks
            << std::endl;

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  const std::string prefill_graph = graphs_to_use[0];
  const std::string generation_graph = graphs_to_use[1];

  auto &prefill_inputs = models[prefill_graph].model_inputs;
  auto &generation_inputs = models[generation_graph].model_inputs;
  auto &prefill_model = models[prefill_graph].model_handle;
  auto &generation_model = models[generation_graph].model_handle;

  auto copy_lookup_embedding = [&](uint16_t *dest, int token_id) {
    const void *embedding = lookupEmbedding(token_id);
    if (embedding == nullptr) {
      throw std::runtime_error("lookupEmbedding(" + std::to_string(token_id) +
                               ") returned null");
    }
    std::memcpy(dest, embedding, bytes_per_token);
  };

  auto fill_generation_embedding_inputs = [&](const void *embedding,
                                              int position) {
    if (position < 0 || position >= rope_cache_seq_len) {
      throw std::runtime_error("Generation position is out of rope cache");
    }

    std::memcpy(generation_sample_u16, embedding, bytes_per_token);

    std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
    std::fill_n(generation_sliding_attention_mask,
                generation_sliding_attention_mask_elements, 0);

    generation_attention_mask[generation_attention_mask_elements - 1] =
        std::numeric_limits<uint16_t>::max();
    generation_sliding_attention_mask
        [generation_sliding_attention_mask_elements - 1] =
            std::numeric_limits<uint16_t>::max();

    for (int i = 0; i < position && i < generation_full_kv_past_length; i++) {
      generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
    }
    for (int i = 0; i < position && i < generation_sliding_kv_past_length;
         i++) {
      generation_sliding_attention_mask[i] =
          std::numeric_limits<uint16_t>::max();
    }

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

  auto prefill_start = std::chrono::high_resolution_clock::now();

  for (unsigned int c = 0; c < n_chunks; c++) {
    const int chunk_offset = c * context_size;
    const int chunk_len =
        ((c + 1) * context_size < input_len)
            ? context_size
            : (static_cast<int>(input_len) - chunk_offset);

    kv_cache_.syncGenerationToPrefill();

    const uint16_t *src_base =
        static_cast<const uint16_t *>(prefill_embeds) +
        static_cast<size_t>(chunk_offset) * hidden_size;
    std::memcpy(input_sample_u16, src_base, chunk_len * bytes_per_token);
    for (int i = chunk_len; i < context_size; i++) {
      copy_lookup_embedding(input_sample_u16 + i * hidden_size, padding_token);
    }

    fill_attention_mask_with_length(context_size, max_seq_len, chunk_len,
                                    attention_mask);
    fill_attention_mask_with_prev_length(
        context_size, max_seq_len,
        std::min(kv_cache_.length(), max_seq_len - context_size),
        attention_mask);

    if (kv_cache_.length() >= generation_sliding_kv_past_length) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window,
                  std::numeric_limits<uint16_t>::min());
      for (int i = 0; i < chunk_len; i++) {
        for (int j = i + 1; j < i + generation_sliding_attention_mask_elements;
             j++) {
          sliding_attention_mask[i * sliding_window + j] =
              std::numeric_limits<uint16_t>::max();
        }
      }
    } else {
      fill_attention_mask_with_length(context_size, sliding_window, chunk_len,
                                      sliding_attention_mask);
      fill_attention_mask_with_prev_length(context_size, sliding_window,
                                           kv_cache_.length(),
                                           sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    if (kv_cache_.length() + chunk_len > rope_cache_seq_len) {
      throw std::runtime_error("Prefill position is out of rope cache");
    }

    const int pos_ids_offset = kv_cache_.length() * pos_dim;
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

    outputs = prefill_model->inference(1, prefill_inputs);
    kv_cache_.appendPrefillOutputs(outputs, kv_cache_.length(), chunk_len,
                                   context_size, prefill_graph);
    kv_cache_.advance(chunk_len);
  }
  auto prefill_end = std::chrono::high_resolution_clock::now();

  LOGD("Generation start...");

  int token = 0;
  auto start = std::chrono::system_clock::now();
  int idx = kv_cache_.length();
  const int prefill_len = kv_cache_.length();
  for (; idx < generation_full_kv_past_length; idx++) {
    if (stop_requested_.load(std::memory_order_acquire)) {
      break;
    }

    if (idx == prefill_len) {
      const uint16_t *last_embed =
          static_cast<const uint16_t *>(prefill_embeds) +
          static_cast<size_t>(input_len) * hidden_size;
      fill_generation_embedding_inputs(last_embed, idx);
    } else {
      const void *embedding = lookupEmbedding(token);
      if (embedding == nullptr) {
        LOGE("run_with_embeddings: lookupEmbedding(%d) null", token);
        break;
      }
      fill_generation_embedding_inputs(embedding, idx);
    }

    outputs = generation_model->inference(1, generation_inputs);
    kv_cache_.appendGenerationOutputs(outputs, idx, 1, 1, generation_graph);
    kv_cache_.advance(1);

    token = sample(std::get<uint16_t *>(outputs[generation_logits_output_index]),
                   vocab_size, seed_tokens.data(), seed_tokens.size(),
                   logit_scale, logit_offset, repetition_penalty, temperature,
                   top_p, top_k);
    LOGD("next_token: %d", token);

    output.push_back(token);
    if (token == eos_token || token == padding_token) {
      break;
    }

    std::string decoded = tokenizer->Decode({token});
    LOGD("%d : %s (idx: %d)", token, decoded.c_str(), idx);
    seed_tokens.push_back(token);
    if (streamer_) {
      if (streamer_put(streamer_, decoded.c_str()) != 0) {
        stop_requested_.store(true, std::memory_order_release);
        break;
      }
    } else if (log_output) {
      std::cout << decoded << std::flush;
    }
  }

  if (streamer_) {
    streamer_end(streamer_);
  }

  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;

  performance_metrics.prefill_tokens = input_len;
  performance_metrics.prefill_duration_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  performance_metrics.generation_tokens = (idx > prefill_len) ? (unsigned int)(idx - prefill_len) : 0U;
  performance_metrics.generation_duration_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  performance_metrics.total_duration_ms =
      performance_metrics.prefill_duration_ms + performance_metrics.generation_duration_ms;
  performance_metrics.peak_memory_kb = 0; // TODO: implement memory tracking

  has_run_ = true;
  if (log_output) {
    const int generated = std::max(1, idx - prefill_len);
    std::cout << std::endl;
    std::cout << std::endl;
    std::cout << "Generation exec_time : " << raw_exec_seconds.count()
              << ", token per second: "
              << generated / raw_exec_seconds.count()
              << ", token generation time average: "
              << raw_exec_seconds.count() / generated << std::endl;
  }
}
