// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_8_qnn.cpp
 * @brief  QNN model implementation with self-registration
 * @note   This model auto-registers with the CausalLM Factory via
 * __attribute__((constructor)) when linked or loaded.
 *
 *         No modification to nntrainer's main.cpp is needed.
 */

#include "gauss3_8_qnn.h"
#include "generate_qnn_utils.h"

#include "api/streamer.h"
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
#include <cctype>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

using namespace causallm;

namespace {

std::string trim_copy(const std::string &value) {
  auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char ch) {
    return std::isspace(ch) != 0;
  });
  auto end =
      std::find_if_not(value.rbegin(), value.rend(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
      }).base();
  if (begin >= end) {
    return "";
  }
  return std::string(begin, end);
}

std::vector<std::string> split_csv_list(const std::string &csv) {
  std::vector<std::string> parts;
  std::stringstream stream(csv);
  std::string item;
  while (std::getline(stream, item, ',')) {
    parts.push_back(trim_copy(item));
  }
  return parts;
}

std::vector<int> parse_dim_string(const std::string &dims) {
  std::vector<int> parsed_dims;
  std::stringstream stream(dims);
  std::string item;
  while (std::getline(stream, item, ':')) {
    parsed_dims.push_back(std::stoi(trim_copy(item)));
  }
  return parsed_dims;
}

int tensor_element_count(const std::vector<int> &dims) {
  int element_count = 1;
  for (int dim : dims) {
    element_count *= dim;
  }
  return element_count;
}

int find_name_index(const std::vector<std::string> &names,
                    const std::string &target_name) {
  for (size_t idx = 0; idx < names.size(); ++idx) {
    if (names[idx] == target_name) {
      return static_cast<int>(idx);
    }
  }
  return -1;
}

int get_named_tensor_elements_or_throw(const std::vector<std::string> &names,
                                       const std::vector<std::string> &dims,
                                       const std::string &target_name) {
  int index = find_name_index(names, target_name);
  if (index < 0 || static_cast<size_t>(index) >= dims.size()) {
    throw std::runtime_error("Missing tensor dims for " + target_name);
  }
  return tensor_element_count(parse_dim_string(dims[index]));
}

int get_named_kv_row_length_or_throw(const std::vector<std::string> &names,
                                     const std::vector<std::string> &dims,
                                     const std::string &target_name) {
  int index = find_name_index(names, target_name);
  if (index < 0 || static_cast<size_t>(index) >= dims.size()) {
    throw std::runtime_error("Missing KV tensor dims for " + target_name);
  }

  auto parsed_dims = parse_dim_string(dims[index]);
  if (parsed_dims.size() < 2) {
    throw std::runtime_error("Unexpected KV dims for " + target_name);
  }
  return parsed_dims.back();
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
      "Gauss_3_8_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                          causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gauss3_8_QNN>(cfg, generation_cfg,
                                                        nntr_cfg);
      });

  // Add more custom models here:
  // causallm::Factory::Instance().registerModel(
  //   "MyNewModelForCausalLM",
  //   [](causallm::json cfg, causallm::json generation_cfg,
  //      causallm::json nntr_cfg) {
  //     return std::make_unique<causallm::MyNewModel>(cfg, generation_cfg,
  //                                                   nntr_cfg);
  //   });
}

void causallm::Gauss3_8_QNN::initialize_input_outputs() {
  auto prefill_output_names_list = split_csv_list(prefill_output_names);
  auto generation_output_names_list = split_csv_list(generation_output_names);

  prefill_hidden_states_output_index =
      find_name_index(prefill_output_names_list, "output_hidden_states");
  generation_hidden_states_output_index =
      find_name_index(generation_output_names_list, "output_hidden_states");

  prefill_attention_mask_elements = get_named_tensor_elements_or_throw(
      prefill_non_embed_input_names, prefill_non_embed_input_dims,
      "attention_mask");
  prefill_sliding_attention_mask_elements = get_named_tensor_elements_or_throw(
      prefill_non_embed_input_names, prefill_non_embed_input_dims,
      "sliding_attention_mask");
  generation_attention_mask_elements = get_named_tensor_elements_or_throw(
      generation_non_embed_input_names, generation_non_embed_input_dims,
      "attention_mask");
  generation_sliding_attention_mask_elements =
      get_named_tensor_elements_or_throw(generation_non_embed_input_names,
                                         generation_non_embed_input_dims,
                                         "sliding_attention_mask");
  generation_full_kv_past_length = generation_attention_mask_elements - 1;
  generation_sliding_kv_past_length =
      generation_sliding_attention_mask_elements - 1;
  rope_cache_seq_len =
      std::max(max_seq_len, generation_attention_mask_elements);

  attention_mask =
      (uint16_t *)tracked_allocate(sizeof(uint16_t) *
                                   prefill_attention_mask_elements);
  sliding_attention_mask =
      (uint16_t *)tracked_allocate(sizeof(uint16_t) *
                                   prefill_sliding_attention_mask_elements);
  generation_attention_mask =
      (uint16_t *)tracked_allocate(sizeof(uint16_t) *
                                   generation_attention_mask_elements);
  generation_sliding_attention_mask =
      (uint16_t *)tracked_allocate(sizeof(uint16_t) *
                                   generation_sliding_attention_mask_elements);

  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);
  allocated_ptrs_.insert(position_ids_cos);
  allocated_ptrs_.insert(position_ids_sin);

  // 확인할 부분 -> sliding window attention?
  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
      get_cos_sin(rope_cache_seq_len, pos_dim, local_rope_theta);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);
  allocated_ptrs_.insert(swa_position_ids_cos);
  allocated_ptrs_.insert(swa_position_ids_sin);

  const int prefill_position_id_elements = get_named_tensor_elements_or_throw(
      prefill_non_embed_input_names, prefill_non_embed_input_dims,
      "position_ids_cos");
  const int generation_position_id_elements = get_named_tensor_elements_or_throw(
      generation_non_embed_input_names, generation_non_embed_input_dims,
      "position_ids_cos");
  const int prefill_swa_position_id_elements =
      get_named_tensor_elements_or_throw(prefill_non_embed_input_names,
                                         prefill_non_embed_input_dims,
                                         "swa_position_ids_cos");
  const int generation_swa_position_id_elements =
      get_named_tensor_elements_or_throw(generation_non_embed_input_names,
                                         generation_non_embed_input_dims,
                                         "swa_position_ids_cos");

  prefill_position_ids_cos = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * prefill_position_id_elements);
  prefill_position_ids_sin = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * prefill_position_id_elements);
  prefill_swa_position_ids_cos = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * prefill_swa_position_id_elements);
  prefill_swa_position_ids_sin = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * prefill_swa_position_id_elements);
  generation_position_ids_cos = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * generation_position_id_elements);
  generation_position_ids_sin = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * generation_position_id_elements);
  generation_swa_position_ids_cos = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * generation_swa_position_id_elements);
  generation_swa_position_ids_sin = (uint16_t *)tracked_allocate(
      sizeof(uint16_t) * generation_swa_position_id_elements);

  std::unordered_map<std::string, ml::train::TensorDim::IO_TensorType>
      prefill_named_inputs;
  std::unordered_map<std::string, ml::train::TensorDim::IO_TensorType>
      generation_named_inputs;

  if (uses_embedding) {
    input_sample = (float *)tracked_allocate(sizeof(float) * context_size);
    generation_sample = (float *)tracked_allocate(sizeof(float));
  } else {
    input_sample_u16 =
        (uint16_t *)tracked_allocate(sizeof(uint16_t) * context_size * hidden_size);
    generation_sample_u16 =
        (uint16_t *)tracked_allocate(sizeof(uint16_t) * hidden_size);
  }

  size_t lora_idx = 0;
  if (lora_path.empty()) {
    for (const auto &name : prefill_non_embed_input_names) {
      if (name.find("_lora_") == std::string::npos) {
        continue;
      }
      if (lora_idx >= lora_sizes.size()) {
        throw std::runtime_error("LoRA size metadata is shorter than input names");
      }
      auto lora_pointer = get_zero_memory(lora_sizes[lora_idx], 32768);
      allocated_ptrs_.insert(lora_pointer);
      prefill_named_inputs[name] = lora_pointer;
      generation_named_inputs[name] = lora_pointer;
      ++lora_idx;
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

    for (const auto &name : prefill_non_embed_input_names) {
      if (name.find("_lora_") == std::string::npos) {
        continue;
      }
      if (lora_idx >= lora_sizes.size()) {
        munmap(mapped, file_size);
        close(fd);
        throw std::runtime_error("LoRA size metadata is shorter than input names");
      }
      auto lora_pointer =
          get_zero_memory(sizeof(uint16_t) * lora_sizes[lora_idx], 0);
      allocated_ptrs_.insert(lora_pointer);
      memcpy(lora_pointer, data_ptr, sizeof(uint16_t) * lora_sizes[lora_idx]);
      prefill_named_inputs[name] = lora_pointer;
      generation_named_inputs[name] = lora_pointer;
      data_ptr += sizeof(uint16_t) * lora_sizes[lora_idx];
      ++lora_idx;
    }

    LOGD("----------------------- initialize() 7");
    munmap(mapped, file_size);
    close(fd);

    std::cout << "LoRA weights loaded from: " << lora_path << std::endl;
  }
  if (lora_idx != lora_sizes.size()) {
    throw std::runtime_error("LoRA input name count does not match lora_sizes");
  }

  this->fresh_kvs.clear();
  this->kvs.clear();
  this->kv_sizes.clear();
  this->kv_row_lengths.clear();

  for (int i = 0; i < num_hidden_layers; i++) {
    const int row_length = get_named_kv_row_length_or_throw(
        generation_non_embed_input_names, generation_non_embed_input_dims,
        "past_key_" + std::to_string(i) + "_h0_in");
    const int size = row_length * head_dim;
    const std::vector<std::string> kv_names = {
        "past_key_" + std::to_string(i) + "_h0_in",
        "past_key_" + std::to_string(i) + "_h1_in",
        "past_value_" + std::to_string(i) + "_h0_in",
        "past_value_" + std::to_string(i) + "_h1_in",
    };
    this->kv_row_lengths.push_back(row_length);

    for (int j = 0; j < 4; j++) {
      int coeff = sizeof(uint16_t) / sizeof(uint8_t);
      this->kv_sizes.push_back(size * coeff);

      // If we cast int8 memory full of 128 to int16, we get 128 * 256 + 128
      // 여기가 prefill kvcache
      auto current_kv = get_zero_memory(size * coeff, 128 * 256 + 128);
      allocated_ptrs_.insert(current_kv);
      prefill_named_inputs[kv_names[j]] = current_kv;
      generation_named_inputs[kv_names[j]] = current_kv;
      this->kvs.push_back(current_kv);
      auto fresh_kv = get_zero_memory(size * coeff, 128 * 256 + 128);
      allocated_ptrs_.insert(fresh_kv);
      this->fresh_kvs.push_back(fresh_kv);
    }
  }

  prefill_named_inputs["attention_mask"] = attention_mask;
  prefill_named_inputs["sliding_attention_mask"] = sliding_attention_mask;
  prefill_named_inputs["position_ids_cos"] = prefill_position_ids_cos;
  prefill_named_inputs["position_ids_sin"] = prefill_position_ids_sin;
  prefill_named_inputs["swa_position_ids_cos"] = prefill_swa_position_ids_cos;
  prefill_named_inputs["swa_position_ids_sin"] = prefill_swa_position_ids_sin;

  generation_named_inputs["attention_mask"] = generation_attention_mask;
  generation_named_inputs["sliding_attention_mask"] =
      generation_sliding_attention_mask;
  generation_named_inputs["position_ids_cos"] = generation_position_ids_cos;
  generation_named_inputs["position_ids_sin"] = generation_position_ids_sin;
  generation_named_inputs["swa_position_ids_cos"] =
      generation_swa_position_ids_cos;
  generation_named_inputs["swa_position_ids_sin"] =
      generation_swa_position_ids_sin;

  // Save pointers to input samples. Type depends on uses_embedding:
  //   uses_embedding=true  -> token-id float* fed into in-graph embedding
  //                           layer (legacy text path).
  //   uses_embedding=false -> pre-computed uint16 embedding vectors fed
  //                           directly as inputs_embeds (multimodal path).
  if (!uses_embedding) {
    // mmap the pre-quantized text embedding table. Both the API-layer
    // composition and the generation loop below call into this via
    // lookupEmbedding() — O(1) per-token access without pulling ~900MB
    // of embeddings into RAM.
    if (!embedding_path.empty()) {
      int fd = ::open(embedding_path.c_str(), O_RDONLY);
      if (fd < 0) {
        LOGE("Gauss3_8_QNN: open embedding file failed: %s",
             embedding_path.c_str());
      } else {
        struct stat st {};
        if (::fstat(fd, &st) == 0) {
          embedding_mmap_size = static_cast<size_t>(st.st_size);
          embedding_mmap_ptr = ::mmap(nullptr, embedding_mmap_size, PROT_READ,
                                      MAP_PRIVATE, fd, 0);
          if (embedding_mmap_ptr == MAP_FAILED) {
            LOGE("Gauss3_8_QNN: mmap embedding file failed");
            embedding_mmap_ptr = nullptr;
            embedding_mmap_size = 0;
          } else {
            (void)::posix_madvise(embedding_mmap_ptr, embedding_mmap_size,
                                  POSIX_MADV_RANDOM);
            embedding_bytes_per_token = hidden_size * sizeof(uint16_t);
            LOGD("Gauss3_8_QNN: embedding table mmap'd (%zu bytes, "
                 "per-token=%zu, vocab≈%zu)",
                 embedding_mmap_size, embedding_bytes_per_token,
                 embedding_mmap_size / embedding_bytes_per_token);
          }
        }
        ::close(fd);
      }
    }
  }

  prefill_inputs.clear();
  generation_inputs.clear();
  if (uses_embedding) {
    prefill_inputs.push_back(input_sample);
    generation_inputs.push_back(generation_sample);
  } else {
    prefill_inputs.push_back(input_sample_u16);
    generation_inputs.push_back(generation_sample_u16);
  }

  for (const auto &name : prefill_non_embed_input_names) {
    auto it = prefill_named_inputs.find(name);
    if (it == prefill_named_inputs.end()) {
      throw std::runtime_error("Missing prefill input binding for " + name);
    }
    prefill_inputs.push_back(it->second);
  }

  for (const auto &name : generation_non_embed_input_names) {
    auto it = generation_named_inputs.find(name);
    if (it == generation_named_inputs.end()) {
      throw std::runtime_error("Missing generation input binding for " + name);
    }
    generation_inputs.push_back(it->second);
  }
}

void causallm::Gauss3_8_QNN::initialize_kv_cache() {
  kv_len = 0;
  
  // KV Cache Initialization
  for (int i = 0; i < this->kvs.size(); i++) {
    std::memcpy(this->kvs[i], this->fresh_kvs[i], this->kv_sizes[i]);
  }
}

void causallm::Gauss3_8_QNN::run(const WSTR prompt, bool do_sample,
                                 const WSTR system_prompt,
                                 const WSTR tail_prompt, bool log_output) {
  auto input = tokenizer->Encode(prompt);

  unsigned int input_len = input.size() - 1;
  if (input_len <= 0) {
    std::cout << "[Error] Input is empty or invalid" << std::endl;
    return;
  }

  auto n_chunks = (input_len % 256 != 0) ? ((input_len / 256) + 1) : (input_len / 256);
  auto token = input.back();

  std::cout << "len: " << input_len << ", n_chunks: " << n_chunks << std::endl;

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  for (int c = 0; c < n_chunks; c++) {
    int chunk_len = ((c + 1) * 256 < input_len) ? context_size : (input_len - (c * 256));
    std::cout << "kv_len: " << kv_len << ", chunk_len: " << chunk_len << std::endl;

    for (int i = 0; i < context_size; i++)
      input_sample[i] = (i < chunk_len) ? input[c * 256 + i] : padding_token;

    fill_attention_mask_with_length(context_size, max_seq_len, chunk_len,
                                    attention_mask);
    fill_attention_mask_with_prev_length(context_size, max_seq_len, kv_len,
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
      fill_attention_mask_with_prev_length(context_size, sliding_window,
                                           kv_len, sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

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

    outputs = prefill_model->inference(1, prefill_inputs);

    if (prefill_hidden_states_output_index >= 0 &&
        prefill_hidden_states_output_index < (int)outputs.size()) {
      outputs.erase(outputs.begin() + prefill_hidden_states_output_index);
    }

    // KV Cache Copy
#pragma omp parallel for
    for (int i = 0; i < (int)this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      int dest_row_length = kv_row_lengths[layer_idx];
      bool is_sliding = dest_row_length == generation_sliding_kv_past_length;

      int src_row_length = 256;
      int num_column = 128;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)this->kvs[kv_idx];

      // Sliding KV wrap — only for sliding layers. Full-context layers
      // have dest_row_length large enough for the whole prefill so they
      // always fall through to the linear write at c * 256.
      int target_idx = kv_len;
      if (is_sliding && kv_len + chunk_len > dest_row_length) {
        target_idx = dest_row_length - chunk_len;
        if (is_key) {
          // K: column-major [num_column][dest_row_length]
          for (int col = 0; col < num_column; ++col) {
            uint8_t *col_base = dest + col * dest_row_length;
            std::memmove(col_base, col_base + chunk_len,
                         dest_row_length - chunk_len);
          }
        } else {
          // V: row-major [dest_row_length][num_column]
          std::memmove(dest, dest + chunk_len * num_column,
                       (dest_row_length - chunk_len) * num_column);
        }
      }

      if (is_key) {
        process_key(output, chunk_len, num_column, dest, target_idx,
                    dest_row_length, src_row_length);
      } else {
        process_value(output, chunk_len, num_column, dest, target_idx);
      }
    };

    kv_len += chunk_len;
  }

  std::fill_n(generation_attention_mask, generation_attention_mask_elements, 0);
  std::fill_n(generation_sliding_attention_mask,
              generation_sliding_attention_mask_elements, 0);

  generation_attention_mask[generation_attention_mask_elements - 1] =
      std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask
      [generation_sliding_attention_mask_elements - 1] =
      std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < kv_len && i < generation_full_kv_past_length; i++)
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < kv_len && i < generation_sliding_kv_past_length; i++)
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = kv_len;
  for (idx = prefill_len; idx < generation_full_kv_past_length; idx++) {
    generation_sample[0] = token;

    generation_attention_mask[idx] = std::numeric_limits<uint16_t>::max();
    if (idx < generation_sliding_kv_past_length)
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

    if (idx > prefill_len) {
      // Remove output_hidden_states from outputs

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
              std::memmove(col_base, col_base + 1, dest_row_length - 1);
            }
          } else {
            std::memmove(dest, dest + num_column,
                         (dest_row_length - 1) * num_column);
          }
        }

        if (is_key) {
          process_key(output, 1, num_column, dest, target_idx, dest_row_length,
                      1);
        } else {
          process_value(output, 1, num_column, dest, target_idx);
        }
      };
    }

    outputs = generation_model->inference(1, generation_inputs);
    if (generation_hidden_states_output_index >= 0 &&
        generation_hidden_states_output_index < (int)outputs.size()) {
      outputs.erase(outputs.begin() + generation_hidden_states_output_index);
    }
    token = sample(std::get<uint16_t *>(outputs.back()), vocab_size,
                   input.data(), input.size(), logit_scale, logit_offset,
                   repetition_penalty, temperature, top_p, top_k);

    output.push_back(token);
    if (token == eos_token) {
      break;
    } else {
      std::string decoded = tokenizer->Decode({token});
      LOGD("%d : %s (idx: %d)", token, decoded.c_str(), idx);
      kv_len += 1;
      // Stream the token if a streamer is attached
      if (streamer_) {
        if (streamer_put(streamer_, decoded.c_str()) != 0) {
          // User requested cancellation
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
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Generation exec_time : " << raw_exec_seconds.count()
            << ", token per second: " << (idx - input_len) / raw_exec_seconds.count()
            << ", token generation time average: "
            << raw_exec_seconds.count() / (idx - input_len) << std::endl;
}

const void *causallm::Gauss3_8_QNN::lookupEmbedding(int token_id) const {
  if (embedding_mmap_ptr == nullptr || embedding_bytes_per_token == 0)
    return nullptr;
  if (token_id < 0)
    return nullptr;
  const size_t offset =
      static_cast<size_t>(token_id) * embedding_bytes_per_token;
  if (offset + embedding_bytes_per_token > embedding_mmap_size)
    return nullptr;
  return static_cast<const uint8_t *>(embedding_mmap_ptr) + offset;
}

std::pair<float, int> causallm::Gauss3_8_QNN::get_embedding_info() {
  return {0.0000029912209811300274, -34952};
}

void causallm::Gauss3_8_QNN::run_with_embeddings(const void *prefill_embeds,
                                                 size_t n_tokens,
                                                 std::vector<int> seed_tokens,
                                                 bool do_sample,
                                                 bool log_output) {

  (void)do_sample;

  if (input_sample_u16 == nullptr || generation_sample_u16 == nullptr) {
    LOGE("run_with_embeddings: u16 input/generation sample not initialized "
         "— was uses_embedding=false set?");
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

  // KV Cache Initialization
  for (int i = 0; i < this->kvs.size(); i++) {
    std::memcpy(this->kvs[i], this->fresh_kvs[i], this->kv_sizes[i]);
  }

  const size_t bytes_per_token = embedding_bytes_per_token;
  const unsigned int _len = static_cast<unsigned int>(n_tokens) - 1;

  auto _n_chunks = (_len % 256 != 0) ? ((_len / 256) + 1) : (_len / 256);

  std::cout << "len: " << _len << ", n_chunks: " << _n_chunks << std::endl;

  std::vector<int> output;
  std::vector<ml::train::TensorDim::IO_TensorType> outputs;

  for (int c = 0; c < _n_chunks; c++) {
    int _chunk_len = ((c + 1) * 256 < _len) ? context_size : (_len - (c * 256));
    int current_kv_len = c * context_size;

    const uint16_t *src_base = static_cast<const uint16_t *>(prefill_embeds) +
                               c * context_size * hidden_size;
    std::memcpy(input_sample_u16, src_base, _chunk_len * bytes_per_token);
    if (_chunk_len < context_size) {
      const void *pad_emb = lookupEmbedding(padding_token);
      if (pad_emb == nullptr) {
        throw std::runtime_error(
            "run_with_embeddings: lookupEmbedding(padding_token=" +
            std::to_string(padding_token) + ") returned null");
      }
      for (int i = _chunk_len; i < context_size; i++) {
        std::memcpy(input_sample_u16 + i * hidden_size, pad_emb,
                    bytes_per_token);
      }
    }

    fill_attention_mask_with_length(context_size, max_seq_len, _chunk_len,
                                    attention_mask);
    fill_attention_mask_with_prev_length(context_size, max_seq_len,
                                         current_kv_len,
                                         attention_mask);

    if (current_kv_len >= generation_sliding_kv_past_length) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window,
                  std::numeric_limits<uint16_t>::min());
      for (int i = 0; i < _chunk_len; i++) {
        for (int j = (i + 1);
             j < (i + generation_sliding_attention_mask_elements); j++) {
          sliding_attention_mask[i * sliding_window + j] =
              std::numeric_limits<uint16_t>::max();
        }
      }
    } else {
      fill_attention_mask_with_length(context_size, sliding_window, _chunk_len,
                                      sliding_attention_mask);
      fill_attention_mask_with_prev_length(context_size, sliding_window,
                                           current_kv_len,
                                           sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    auto pos_ids_offset = current_kv_len * pos_dim;
    std::memcpy(prefill_position_ids_cos, position_ids_cos + pos_ids_offset,
                _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_position_ids_sin, position_ids_sin + pos_ids_offset,
                _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_cos,
                swa_position_ids_cos + pos_ids_offset,
                _chunk_len * pos_dim * sizeof(uint16_t));
    std::memcpy(prefill_swa_position_ids_sin,
                swa_position_ids_sin + pos_ids_offset,
                _chunk_len * pos_dim * sizeof(uint16_t));

    outputs = prefill_model->inference(1, prefill_inputs);

    if (prefill_hidden_states_output_index >= 0 &&
        prefill_hidden_states_output_index < (int)outputs.size()) {
      outputs.erase(outputs.begin() + prefill_hidden_states_output_index);
    }

    // KV Cache Copy
#pragma omp parallel for
    for (int i = 0; i < (int)this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      int dest_row_length = kv_row_lengths[layer_idx];
      bool is_sliding = dest_row_length == generation_sliding_kv_past_length;

      int src_row_length = 256;
      int num_column = 128;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)this->kvs[kv_idx];

      // Sliding KV wrap — only for sliding layers. Full-context layers
      // have dest_row_length large enough for the whole prefill so they
      // always fall through to the linear write at c * 256.
      int target_idx = current_kv_len;
      if (is_sliding && current_kv_len + _chunk_len > dest_row_length) {
        target_idx = dest_row_length - _chunk_len;
        if (is_key) {
          // K: column-major [num_column][dest_row_length]
          for (int col = 0; col < num_column; ++col) {
            uint8_t *col_base = dest + col * dest_row_length;
            std::memmove(col_base, col_base + _chunk_len,
                         dest_row_length - _chunk_len);
          }
        } else {
          // V: row-major [dest_row_length][num_column]
          std::memmove(dest, dest + _chunk_len * num_column,
                       (dest_row_length - _chunk_len) * num_column);
        }
      }

      if (is_key) {
        process_key(output, _chunk_len, num_column, dest, target_idx,
                    dest_row_length, src_row_length);
      } else {
        process_value(output, _chunk_len, num_column, dest, target_idx);
      }
    };
  }

  LOGD("Generation start...");

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
  for (int i = 0; i < _len && i < generation_sliding_kv_past_length; i++)
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  int token = 0;

  auto start = std::chrono::system_clock::now();
  int idx;
  for (idx = _len; idx < generation_full_kv_past_length; idx++) {
    if (idx == _len) {
      // First gen iter: use the LAST prefill embedding (position _len),
      // which was deliberately excluded from the prefill batch — the
      // same role _input.back() plays in the text run() path.
      const uint16_t *last_embed =
          static_cast<const uint16_t *>(prefill_embeds) + _len * hidden_size;
      std::memcpy(generation_sample_u16, last_embed, bytes_per_token);
    } else {
      const void *emb = lookupEmbedding(token);
      if (emb == nullptr) {
        LOGE("run_with_embeddings: lookupEmbedding(%d) null", token);
        break;
      }
      std::memcpy(generation_sample_u16, emb, bytes_per_token);
    }

    generation_attention_mask[idx] = std::numeric_limits<uint16_t>::max();
    if (idx < generation_sliding_kv_past_length)
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

    if (idx > _len) {
      // Remove output_hidden_states from outputs

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
              std::memmove(col_base, col_base + 1, dest_row_length - 1);
            }
          } else {
            std::memmove(dest, dest + num_column,
                         (dest_row_length - 1) * num_column);
          }
        }

        if (is_key) {
          process_key(output, 1, num_column, dest, target_idx, dest_row_length,
                      1);
        } else {
          process_value(output, 1, num_column, dest, target_idx);
        }
      };
    }

    outputs = generation_model->inference(1, generation_inputs);
    if (generation_hidden_states_output_index >= 0 &&
        generation_hidden_states_output_index < (int)outputs.size()) {
      outputs.erase(outputs.begin() + generation_hidden_states_output_index);
    }
    LOGD("After generation_model->inference");
    int next_token =
        sample(std::get<uint16_t *>(outputs.back()), vocab_size,
               seed_tokens.data(), seed_tokens.size(), logit_scale,
               logit_offset, repetition_penalty, temperature, top_p, top_k);
    LOGD("next_token: %d", next_token);
    token = next_token;

    if (next_token == eos_token) {
      break;
    } else {
      std::string decoded = tokenizer->Decode({token});
      LOGD("%d : %s (idx: %d)", token, decoded.c_str(), idx);
      // Stream the token if a streamer is attached
      if (streamer_) {
        if (streamer_put(streamer_, decoded.c_str()) != 0) {
          // User requested cancellation
          break;
        }
      } else if (log_output) {
        std::cout << decoded << std::flush;
      }
    }
  }

  // Notify the streamer that generation is complete
  if (streamer_) {
    streamer_end(streamer_);
  }

  has_run_ = true;
  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Generation exec_time : " << raw_exec_seconds.count()
            << ", token per second: " << (idx - _len) / raw_exec_seconds.count()
            << ", token generation time average: "
            << raw_exec_seconds.count() / (idx - _len) << std::endl;
}
