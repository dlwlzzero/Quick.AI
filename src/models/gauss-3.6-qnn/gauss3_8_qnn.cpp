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

#include <llm_util.hpp>
#include <model.h>

#include <app_context.h>
#include <engine.h>
#include <factory.h>

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
      get_cos_sin(context_size, pos_dim, rope_theta);
  position_ids_cos = std::get<0>(cos_sin_tuple);
  position_ids_sin = std::get<1>(cos_sin_tuple);

  std::tuple<uint16_t *, uint16_t *> swa_cos_sin_tuple =
      get_cos_sin(context_size, pos_dim, local_rope_theta);
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

  input_sample = (float *)allocate(sizeof(float) * (context_size));
  generation_sample = (float *)allocate(sizeof(float));

  prefill_inputs = {input_sample};
  generation_inputs = {generation_sample};

  // Zero lora values. This can be read from file later
  for (int i = 0; i < lora_sizes.size(); i++) {
    auto lora_pointer = get_zero_memory(lora_sizes[i], 32768);
    prefill_inputs.push_back(lora_pointer);
    generation_inputs.push_back(lora_pointer);
  }

  prefill_inputs.push_back(prefill_swa_position_ids_cos);
  generation_inputs.push_back(generation_swa_position_ids_cos);
  prefill_inputs.push_back(prefill_swa_position_ids_sin);
  generation_inputs.push_back(generation_swa_position_ids_sin);

  this->fresh_kvs.clear();
  this->kvs.clear();
  this->kv_sizes.clear();
  for (int i = 0; i < num_hidden_layers; i++) {
    int attn_length;
    int generation_attn_length;
    if (i % 5 == 4) {
      // no sliding attn
      attn_length = max_seq_len;
      generation_attn_length = max_seq_len - 1;
    } else {
      attn_length = sliding_window;
      generation_attn_length = sliding_window - context_size - 1;
    }
    int prefill_size = attn_length * head_dim;
    int size = generation_attn_length * head_dim;
    // key, value, first head, second head
    for (int j = 0; j < 4; j++) {
      int coeff = sizeof(uint16_t) / sizeof(uint8_t);
      this->kv_sizes.push_back(size * coeff);
      if (i < max_window_layers) {
        // If we cast int8 memory full of 128 to int16, we get 128 * 256 + 128
        prefill_inputs.push_back(
            get_zero_memory(prefill_size * coeff, 128 * 256 + 128));
      }
      auto current_kv = get_zero_memory(size * coeff, 128 * 256 + 128);
      generation_inputs.push_back(current_kv);
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
}

void causallm::Gauss3_8_QNN::run(const WSTR prompt, bool do_sample,
                                 const WSTR system_prompt,
                                 const WSTR tail_prompt, bool log_output) {
  auto _input = tokenizer->Encode(prompt);
  unsigned int _len = _input.size() - 1;
  for (int i = 0; i < context_size; i++)
    input_sample[i] = padding_token;
  std::cout << "input length: " << _len << std::endl;
  for (int i = 0; i < _len; i++) {
    std::cout << _input[i] << ", ";
    input_sample[i] = _input[i];
  }
  std::cout << _input[_len] << std::endl;

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

  std::cout << "before prefill model run..." << std::endl;
  auto outputs = prefill_model->inference(1, prefill_inputs);
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
#pragma omp parallel for
    for (int i = 0; i < this->kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      int dest_row_length = layer_idx % 5 == 4
                                ? max_seq_len - 1
                                : sliding_window - context_size - 1;
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
    if (token == eos_token) {
      std::cout << "Finished generating, break..." << std::endl;
      break;
    } else {
      std::cout << tokenizer->Decode({token}) << std::flush;
      _input.push_back(token);
    }
  }
  auto end = std::chrono::system_clock::now();
  raw_exec_seconds = end - start;
  std::cout << std::endl;
  std::cout << std::endl;
  std::cout << "Generation exec_time : " << raw_exec_seconds.count()
            << ", token per second: " << (idx - _len) / raw_exec_seconds.count()
            << ", token generation time average: "
            << raw_exec_seconds.count() / (idx - _len) << std::endl;
}