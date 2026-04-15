#include "quick_dot_ai_qnn.h"
#include "engine.h"
#include "generate_qnn_utils.h"

#include <iostream>
#include <android/log.h>

#define LOG_TAG "Quick_Dot_AI_QNN"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace ml::train;
using namespace nntrainer;

void causallm::Quick_Dot_AI_QNN::initialize() {
  // Call the overloaded version with empty native_lib_dir
  initialize("");
}

void causallm::Quick_Dot_AI_QNN::initialize(const std::string &native_lib_dir) {
  int status;
  int non_embed_input_count;

  auto &ct_engine = nntrainer::Engine::Global();

  LOGD("%s:%d", __FILE__, __LINE__);
  LOGD("native_lib_dir: %s", native_lib_dir.c_str());

  NNTR_THROW_IF(ct_engine.registerContext("libqnn_context.so", native_lib_dir),
                std::runtime_error)
      << "Fail to register QNN Context";

        LOGD("%s:%d", __FILE__, __LINE__);


  prefill_model = createModel(ml::train::ModelType::NEURAL_NET);

    LOGD("%s:%d", __FILE__, __LINE__);


  prefill_model->addLayer(createLayer(
      "embedding",
      {withKey("name", "inputs_embeds"), withKey("in_dim", vocab_size),
       withKey("input_shape", "1:" + std::to_string(sequence_length)),
       withKey("out_dim", hidden_size)}));

         LOGD("%s:%d", __FILE__, __LINE__);


  NNTR_THROW_IF(prefill_non_embed_input_names.size() !=
                    prefill_non_embed_input_dims.size(),
                std::invalid_argument)
      << "Non-embedding input names and dimensions should have equal size";

        LOGD("%s:%d", __FILE__, __LINE__);

  non_embed_input_count = prefill_non_embed_input_names.size();

    LOGD("%s:%d", __FILE__, __LINE__);

  for (int i = 0; i < non_embed_input_count; i++) {
    prefill_model->addLayer(createLayer(
        "input", {withKey("name", prefill_non_embed_input_names[i]),
                  withKey("input_shape", prefill_non_embed_input_dims[i])}));
  }

    LOGD("%s:%d", __FILE__, __LINE__);


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

                     LOGD("%s:%d", __FILE__, __LINE__);


  prefill_model->addLayer(prefill_qnn_layer);

    LOGD("%s:%d", __FILE__, __LINE__);


  prefill_model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                              withKey("model_tensor_type", "UINT16-UINT16")});

                                LOGD("%s:%d", __FILE__, __LINE__);


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

  // TODO can this line be moved to initialize() ?
  initialize_input_outputs();
}

void causallm::Quick_Dot_AI_QNN::save_weight(const std::string &weight_path) {
  // Unimplemented.
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
