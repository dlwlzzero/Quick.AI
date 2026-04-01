#include "quick_dot_ai_qnn.h"
#include "engine.h"
#include "generate_qnn_utils.hpp"

#include <iostream>

using namespace ml::train;
using namespace nntrainer;

void causallm::Quick_Dot_AI_QNN::initialize() {
  int status;
  int non_embed_input_count;

  auto &ct_engine = nntrainer::Engine::Global();

  NNTR_THROW_IF(ct_engine.registerContext("libqnn_context.so", ""),
                std::runtime_error)
      << "Fail to register QNN Context";

  prefill_model = createModel(ml::train::ModelType::NEURAL_NET);

  prefill_model->addLayer(createLayer(
      "embedding",
      {withKey("name", "inputs_embeds"), withKey("in_dim", vocab_size),
       withKey("input_shape", "1:" + std::to_string(sequence_length)),
       withKey("out_dim", hidden_size)}));

  NNTR_THROW_IF(prefill_non_embed_input_names.size() !=
                    prefill_non_embed_input_dims.size(),
                std::invalid_argument)
      << "Non-embedding input names and dimensions should have equal size";
  non_embed_input_count = prefill_non_embed_input_names.size();
  for (int i = 0; i < non_embed_input_count; i++) {
    prefill_model->addLayer(createLayer(
        "input", {withKey("name", prefill_non_embed_input_names[i]),
                  withKey("input_shape", prefill_non_embed_input_dims[i])}));
  }

  LayerHandle prefill_qnn_layer =
      createLayer("qnn_graph",
                  {withKey("name", prefill_graph_name),
                   withKey("path", model_path), withKey("dim", prefill_out_dim),
                   withKey("tensor_dtype", prefill_out_data_format),
                   withKey("tensor_type", prefill_out_tensor_format),
                   withKey("input_layers", prefill_input_names),
                   withKey("input_quant_param", prefill_in_quant),
                   withKey("output_quant_param", prefill_out_quant),
                   withKey("engine", "qnn")});
  prefill_model->addLayer(prefill_qnn_layer);

  prefill_model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                              withKey("model_tensor_type", "UINT16-UINT16")});

  auto prefill_optimizer =
      createOptimizer("sgd", {withKey("learning_rate", 0.001)});
  prefill_model->setOptimizer(std::move(prefill_optimizer));

  status = prefill_model->compile(ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("Prefill model compilation failed!");
  }

  status = prefill_model->initialize(ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("Prefill model initialization failed!");
  }

  generation_model = createModel(ml::train::ModelType::NEURAL_NET);

  generation_model->addLayer(createLayer(
      "embedding",
      {withKey("name", "inputs_embeds"), withKey("in_dim", vocab_size),
       withKey("input_shape", "1:1"), withKey("out_dim", hidden_size)}));

  NNTR_THROW_IF(generation_non_embed_input_names.size() !=
                    generation_non_embed_input_dims.size(),
                std::invalid_argument)
      << "Non-embedding input names and dimensions should have equal size";
  non_embed_input_count = generation_non_embed_input_names.size();
  for (int i = 0; i < non_embed_input_count; i++) {
    generation_model->addLayer(createLayer(
        "input", {withKey("name", generation_non_embed_input_names[i]),
                  withKey("input_shape", generation_non_embed_input_dims[i])}));
  }

  LayerHandle generation_qnn_layer = createLayer(
      "qnn_graph",
      {withKey("name", generation_graph_name), withKey("path", model_path),
       withKey("dim", generation_out_dim),
       withKey("tensor_dtype", generation_out_data_format),
       withKey("tensor_type", generation_out_tensor_format),
       withKey("input_layers", generation_input_names),
       withKey("input_quant_param", generation_in_quant),
       withKey("output_quant_param", generation_out_quant),
       withKey("engine", "qnn")});
  generation_model->addLayer(generation_qnn_layer);

  generation_model->setProperty(
      {withKey("batch_size", 1), withKey("epochs", 1),
       withKey("model_tensor_type", "UINT16-UINT16")});

  auto generation_optimizer =
      createOptimizer("sgd", {withKey("learning_rate", 0.001)});
  generation_model->setOptimizer(std::move(generation_optimizer));

  status = generation_model->compile(ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("Prefill model compilation failed!");
  }

  status = generation_model->initialize(ExecutionMode::INFERENCE);
  if (status) {
    throw std::invalid_argument("Prefill model initialization failed!");
  }

  // TODO check tokenizer initialization after API change
}

void causallm::Quick_Dot_AI_QNN::load_weight(const std::string &weight_path) {
  prefill_model->load(model_path, ModelFormat::MODEL_FORMAT_QNN);
  prefill_model->load(embedding_path);
  prefill_model->allocate();

  generation_model->load(model_path, ModelFormat::MODEL_FORMAT_QNN);
  generation_model->load(embedding_path);
  generation_model->allocate();
}

void causallm::Quick_Dot_AI_QNN::save_weight(const std::string &weight_path) {
  // Unimplemented.
}

void causallm::Quick_Dot_AI_QNN::run(const WSTR prompt, bool do_sample,
                                     const WSTR system_prompt,
                                     const WSTR tail_prompt, bool log_output) {
  auto attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * max_seq_len);
  auto sliding_attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * sliding_window);
  auto generation_attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * max_seq_len);
  auto generation_sliding_attention_mask =
      (uint16_t *)allocate(sizeof(uint16_t) * (sliding_window - context_size));

  auto [position_ids_cos, position_ids_sin] =
      get_cos_sin(context_size, pos_dim, rope_theta);
  auto [swa_position_ids_cos, swa_position_ids_sin] =
      get_cos_sin(context_size, pos_dim, local_rope_theta);
  auto prefill_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  auto prefill_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  auto prefill_swa_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  auto prefill_swa_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * context_size * pos_dim);
  auto generation_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);
  auto generation_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);
  auto generation_swa_position_ids_cos =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);
  auto generation_swa_position_ids_sin =
      (uint16_t *)allocate(sizeof(uint16_t) * pos_dim);

  float *input_sample = (float *)allocate(sizeof(float) * (context_size));
  float *generation_sample = (float *)allocate(sizeof(float));
  std::vector<IO_TensorType> prefill_inputs = {input_sample,
                                               attention_mask,
                                               sliding_attention_mask,
                                               prefill_position_ids_cos,
                                               prefill_position_ids_sin,
                                               prefill_swa_position_ids_cos,
                                               prefill_swa_position_ids_sin};
  std::vector<IO_TensorType> generation_inputs = {
      generation_sample,
      generation_attention_mask,
      generation_sliding_attention_mask,
      generation_position_ids_cos,
      generation_position_ids_sin,
      generation_swa_position_ids_cos,
      generation_swa_position_ids_sin,
  };

  // Zero lora values. This can be read from file later
  for (int i = 0; i < lora_sizes.size(); i++) {
    auto lora_pointer = get_zero_memory(lora_sizes[i], 32768);
    prefill_inputs.push_back(lora_pointer);
    generation_inputs.push_back(lora_pointer);
  }

  std::vector<uint16_t *> fresh_kvs;
  std::vector<uint16_t *> kvs;
  std::vector<int> kv_sizes;
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
      kv_sizes.push_back(size * coeff);
      if (i < max_window_layers) {
        // If we cast int8 memory full of 128 to int16, we get 128 * 256 + 128
        prefill_inputs.push_back(
            get_zero_memory(prefill_size * coeff, 128 * 256 + 128));
      }
      auto current_kv = get_zero_memory(size * coeff, 128 * 256 + 128);
      generation_inputs.push_back(current_kv);
      kvs.push_back(current_kv);
      fresh_kvs.push_back(get_zero_memory(size * coeff, 128 * 256 + 128));
    }
  }

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

  for (int i = 0; i < kvs.size(); i++) {
    std::memcpy(kvs[i], fresh_kvs[i], kv_sizes[i]);
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
    for (int i = 0; i < kvs.size(); i++) {
      bool is_key = i % 4 > 1;
      int kv_idx = i / 4 * 4 + (i + 2) % 4;
      int layer_idx = i / 4;
      int dest_row_length = layer_idx % 5 == 4
                                ? max_seq_len - 1
                                : sliding_window - context_size - 1;
      int src_row_length = idx == _len ? 256 : 1;

      auto output = std::get<uint8_t *>(outputs[i]);
      auto dest = (uint8_t *)kvs[kv_idx];
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

void causallm::Quick_Dot_AI_QNN::setupParameters(json &cfg,
                                                 json &generation_cfg,
                                                 json &nntr_cfg) {
  // Read nntr_config parameters
  model_path = nntr_cfg["model_file_name"].get<std::string>();
  embedding_path = nntr_cfg["embedding_file_name"].get<std::string>();
  tokenizer_path = nntr_cfg["tokenizer_file"].get<std::string>();

  // Read config parameters - prefill graph
  prefill_graph_name = cfg["prefill_graph_name"].get<std::string>();
  prefill_input_names = cfg["prefill_input_names"].get<std::string>();
  prefill_output_names = cfg["prefill_output_names"].get<std::string>();
  prefill_in_quant = cfg["prefill_in_quant"].get<std::string>();
  prefill_out_quant = cfg["prefill_out_quant"].get<std::string>();
  prefill_in_dim = cfg["prefill_in_dim"].get<std::string>();
  prefill_out_dim = cfg["prefill_out_dim"].get<std::string>();
  prefill_in_data_format = cfg["prefill_in_data_format"].get<std::string>();
  prefill_out_data_format = cfg["prefill_out_data_format"].get<std::string>();
  prefill_out_tensor_format =
      cfg["prefill_out_tensor_format"].get<std::string>();
  prefill_non_embed_input_names =
      cfg["prefill_non_embed_input_names"].get<std::vector<std::string>>();
  prefill_non_embed_input_dims =
      cfg["prefill_non_embed_input_dims"].get<std::vector<std::string>>();

  // Read config parameters - generation graph
  generation_graph_name = cfg["generation_graph_name"].get<std::string>();
  generation_input_names = cfg["generation_input_names"].get<std::string>();
  generation_output_names = cfg["generation_output_names"].get<std::string>();
  generation_in_quant = cfg["generation_in_quant"].get<std::string>();
  generation_out_quant = cfg["generation_out_quant"].get<std::string>();
  generation_in_dim = cfg["generation_in_dim"].get<std::string>();
  generation_out_dim = cfg["generation_out_dim"].get<std::string>();
  generation_in_data_format =
      cfg["generation_in_data_format"].get<std::string>();
  generation_out_data_format =
      cfg["generation_out_data_format"].get<std::string>();
  generation_out_tensor_format =
      cfg["generation_out_tensor_format"].get<std::string>();
  generation_non_embed_input_names =
      cfg["generation_non_embed_input_names"].get<std::vector<std::string>>();
  generation_non_embed_input_dims =
      cfg["generation_non_embed_input_dims"].get<std::vector<std::string>>();

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
  lora_sizes = cfg["lora_sizes"].get<std::vector<int>>();

  // Read generation_config parameters
  padding_token = generation_cfg["padding_token"].get<int>();
  eos_token = generation_cfg["eos_token_id"].get<int>();
  temperature = generation_cfg["temperature"].get<float>();
  top_k = generation_cfg["top_k"].get<int>();
  top_p = generation_cfg["top_p"].get<float>();
  repetition_penalty = generation_cfg["repetition_penalty"].get<float>();
  logit_scale = generation_cfg["logit_scale"].get<float>();
  logit_offset = generation_cfg["logit_offset"].get<int>();
}

void causallm::Quick_Dot_AI_QNN::constructModel() {
  // Unimplemented.
}

std::vector<causallm::LayerHandle>
causallm::Quick_Dot_AI_QNN::createTransformerDecoderBlock(
    const int layer_id, std::string input_name) {
  // Unimplemented.
  return std::vector<LayerHandle>();
}

std::vector<causallm::LayerHandle> causallm::Quick_Dot_AI_QNN::createAttention(
    const int layer_id, int sequence_length, int n_heads, int head_dim,
    std::string query_name, std::string key_name, std::string value_name) {
  // Unimplemented.
  return std::vector<LayerHandle>();
}

std::vector<causallm::LayerHandle>
causallm::Quick_Dot_AI_QNN::createMlp(const int layer_id, int dim,
                                      int hidden_dim, std::string input_name) {
  // Unimplemented.
  return std::vector<LayerHandle>();
}

void causallm::Quick_Dot_AI_QNN::registerCustomLayers() {
  // Unimplemented.
}
