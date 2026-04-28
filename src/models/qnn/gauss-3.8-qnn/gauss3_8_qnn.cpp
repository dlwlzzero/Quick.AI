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

#include <iostream>

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
  attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * max_seq_len);
  sliding_attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * sliding_window);
  generation_attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * max_seq_len);
  generation_sliding_attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * (sliding_window - context_size));

  std::tuple<uint16_t *, uint16_t *> cos_sin_tuple =
      get_cos_sin(max_seq_len, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);

  // 확인할 부분 -> sliding window attention?
  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
      get_cos_sin(max_seq_len, pos_dim, local_rope_theta);
  swa_position_ids_cos = std::get<0>(swa_cos_sin_tuple);
  swa_position_ids_sin = std::get<1>(swa_cos_sin_tuple);

  prefill_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  prefill_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  prefill_swa_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  prefill_swa_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  generation_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);
  generation_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);
  generation_swa_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);
  generation_swa_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);

  if (uses_embedding) {
    input_sample = (float *)allocate(sizeof(float) * context_size);
    generation_sample = (float *)allocate(sizeof(float));
    prefill_inputs = {input_sample};
    generation_inputs = {generation_sample};
  } else {
    input_sample_u16 =
        (uint16_t *)allocate(sizeof(uint16_t) * context_size * hidden_size);
    generation_sample_u16 =
        (uint16_t *)allocate(sizeof(uint16_t) * hidden_size);
    prefill_inputs = {input_sample_u16};
    generation_inputs = {generation_sample_u16};
  }

  // Initialize LoRA tensors
  if (lora_path.empty()) {
    // Default: fill with 32768 (zero value for quantized uint16_t)
    for (int i = 0; i < lora_sizes.size(); i++) {
      auto lora_pointer = get_zero_memory(lora_sizes[i], 32768);
      prefill_inputs.push_back(lora_pointer);
      generation_inputs.push_back(lora_pointer);
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

    for (int i = 0; i < lora_sizes.size(); i++) {
      auto lora_pointer = get_zero_memory(sizeof(uint16_t) * lora_sizes[i], 0);
      memcpy(lora_pointer, data_ptr, sizeof(uint16_t) * lora_sizes[i]);
      prefill_inputs.push_back(lora_pointer);
      generation_inputs.push_back(lora_pointer);
      data_ptr += sizeof(uint16_t) * lora_sizes[i];
    }

    LOGD("----------------------- initialize() 7");
    munmap(mapped, file_size);
    close(fd);

    std::cout << "LoRA weights loaded from: " << lora_path << std::endl;
  }

  prefill_inputs.push_back(prefill_swa_position_ids_cos);
  generation_inputs.push_back(generation_swa_position_ids_cos);
  prefill_inputs.push_back(prefill_swa_position_ids_sin);
  generation_inputs.push_back(generation_swa_position_ids_sin);

  this->fresh_kvs.clear();
  this->kvs.clear();
  this->kv_sizes.clear();

  for (int i = 0; i < num_hidden_layers; i++) {
    int attn_length = (i % 5 == 4) ? max_seq_len - context_size
                                   : sliding_window - context_size;
    int size = attn_length * head_dim;
    // key, value, first head, second head
    for (int j = 0; j < 4; j++) {
      int coeff = sizeof(uint16_t) / sizeof(uint8_t);
      this->kv_sizes.push_back(size * coeff);

      // If we cast int8 memory full of 128 to int16, we get 128 * 256 + 128
      // 여기가 prefill kvcache
      auto current_kv = get_zero_memory(size * coeff, 128 * 256 + 128);
      generation_inputs.push_back(current_kv);
      prefill_inputs.push_back(current_kv);
      this->kvs.push_back(current_kv);
      this->fresh_kvs.push_back(get_zero_memory(size * coeff, 128 * 256 + 128));
    }

    if (i == 0) {
      prefill_inputs.push_back(sliding_attention_mask);
      generation_inputs.push_back(generation_sliding_attention_mask);
    }

    else if (i == 3) {
      prefill_inputs.push_back(prefill_position_ids_cos);
      generation_inputs.push_back(generation_position_ids_cos);
      prefill_inputs.push_back(prefill_position_ids_sin);
      generation_inputs.push_back(generation_position_ids_sin);
    }

    else if (i == 4) {
      prefill_inputs.push_back(attention_mask);
      generation_inputs.push_back(generation_attention_mask);
    }
  }

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

    if (kv_len >= 1024) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window,
                  std::numeric_limits<uint16_t>::min());
      for (int i = 0; i < chunk_len; i++) {
        for (int j = (i + 1); j < (i + 1025); j++) {
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

    // Remove output_hidden_states from outputs
    outputs.erase(outputs.begin() + 96);

    // KV Cache Copy
#pragma omp parallel for
    for (int i = 0; i < (int)this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      bool is_sliding = (layer_idx % 5 != 4);
      int dest_row_length = is_sliding ? sliding_window - context_size
                                       : max_seq_len - context_size;

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

  std::fill_n(generation_attention_mask, max_seq_len, 0);
  std::fill_n(generation_sliding_attention_mask, sliding_window - context_size,
              0);

  generation_attention_mask[max_seq_len - 1] =
      std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask[(sliding_window - context_size) - 1] =
      std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < kv_len; i++)
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < kv_len && i < sliding_window - context_size; i++)
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  auto start = std::chrono::system_clock::now();
  int idx;
  int prefill_len = kv_len;
  for (idx = prefill_len; idx < (max_seq_len - context_size); idx++) {
    generation_sample[0] = token;

    generation_attention_mask[idx] = std::numeric_limits<uint16_t>::max();
    if (idx < sliding_window - context_size)
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
        bool is_sliding = (layer_idx % 5 != 4);
        int dest_row_length = layer_idx % 5 == 4
                                  ? max_seq_len - context_size
                                  : sliding_window - context_size;
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
    outputs.erase(outputs.begin() + 96);
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
    fill_attention_mask_with_prev_length(context_size, max_seq_len, c * 256,
                                         attention_mask);

    if (c >= 4) {
      std::fill_n(sliding_attention_mask, context_size * sliding_window,
                  std::numeric_limits<uint16_t>::min());
      for (int i = 0; i < _chunk_len; i++) {
        for (int j = (i + 1); j < (i + 1025); j++) {
          sliding_attention_mask[i * sliding_window + j] =
              std::numeric_limits<uint16_t>::max();
        }
      }
    } else {
      fill_attention_mask_with_length(context_size, sliding_window, _chunk_len,
                                      sliding_attention_mask);
      fill_attention_mask_with_prev_length(context_size, sliding_window,
                                           c * 256, sliding_attention_mask);
    }

    std::fill_n(prefill_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_position_ids_sin, context_size * pos_dim, 32768);
    std::fill_n(prefill_swa_position_ids_cos, context_size * pos_dim, 65535);
    std::fill_n(prefill_swa_position_ids_sin, context_size * pos_dim, 32768);

    auto pos_ids_offset = c * context_size * pos_dim;
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

    // Remove output_hidden_states from outputs
    outputs.erase(outputs.begin() + 96);

    // KV Cache Copy
#pragma omp parallel for
    for (int i = 0; i < (int)this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      bool is_sliding = (layer_idx % 5 != 4);
      int dest_row_length = is_sliding ? sliding_window - context_size
                                       : max_seq_len - context_size;

      int src_row_length = 256;
      int num_column = 128;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)this->kvs[kv_idx];

      // Sliding KV wrap — only for sliding layers. Full-context layers
      // have dest_row_length large enough for the whole prefill so they
      // always fall through to the linear write at c * 256.
      int target_idx = c * 256;
      if (is_sliding && c * 256 + _chunk_len > dest_row_length) {
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

  std::fill_n(generation_attention_mask, max_seq_len, 0);
  std::fill_n(generation_sliding_attention_mask, sliding_window - context_size,
              0);

  generation_attention_mask[max_seq_len - 1] =
      std::numeric_limits<uint16_t>::max();
  generation_sliding_attention_mask[(sliding_window - context_size) - 1] =
      std::numeric_limits<uint16_t>::max();

  for (int i = 0; i < _len; i++)
    generation_attention_mask[i] = std::numeric_limits<uint16_t>::max();
  for (int i = 0; i < _len && i < sliding_window - context_size; i++)
    generation_sliding_attention_mask[i] = std::numeric_limits<uint16_t>::max();

  int token = 0;

  auto start = std::chrono::system_clock::now();
  int idx;
  for (idx = _len; idx < (max_seq_len - context_size); idx++) {
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
    if (idx < sliding_window - context_size)
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
        bool is_sliding = (layer_idx % 5 != 4);
        int dest_row_length = layer_idx % 5 == 4
                                  ? max_seq_len - context_size
                                  : sliding_window - context_size;
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
    outputs.erase(outputs.begin() + 96);
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
