#include "quick_dot_ai_qnn.h"
#include "android_memory_allocator.h"
#include "engine.h"
#include "graph_parser.h"

using namespace ml::train;
using namespace nntrainer;
using namespace causallm;

std::string qnn_to_nntrainer_datatype(std::string qnn_dtype) {
  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16") {
    return "UINT16";
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    return "UINT8";
  } else {
    throw std::invalid_argument("qnn_dtype is " + qnn_dtype);
  }
}

ml::train::TensorDim::IO_TensorType
get_qnn_input_data(TensorInfo tensor_object) {
  int size = GraphParser::get_tensor_size(tensor_object);
  std::string qnn_dtype = tensor_object.data_type;

  if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_16") {
    return (uint16_t *)allocate(size);
  } else if (qnn_dtype == "QNN_DATATYPE_UFIXED_POINT_8") {
    return (uint8_t *)allocate(size);
  } else {
    throw std::invalid_argument("qnn_dtype is " + qnn_dtype);
  }
}

causallm::Quick_Dot_AI_QNN::~Quick_Dot_AI_QNN() {
  for(auto &[model_name, model]:models){
    model.model_handle.reset();
  }
  for (const auto &[model_name, model] : models) {
    for (const auto &input : model.model_inputs) {
      void *ptr = std::visit(
          [](auto *p) -> void * { return static_cast<void *>(p); }, input);
      deallocate(ptr);
    }
  }
}

void causallm::Quick_Dot_AI_QNN::initialize() {
  int status;

  auto &ct_engine = nntrainer::Engine::Global();
  LOGD("qnn_engine registering .... ");

  NNTR_THROW_IF(ct_engine.registerContext("libqnn_context.so", ""),
                std::runtime_error)
    << "Fail to register QNN Context";

  LOGD("qnn_engine registering done "); 

  GraphParser graph_parser = GraphParser();
  auto graphs_info = graph_parser.parseJsonFile(binary_config_path);
  for (const auto &graph_name : graphs_to_use) {
    auto current_model = createModel(ml::train::ModelType::NEURAL_NET);
    std::string out_dim;
    std::string out_data_format;
    std::string out_tensor_format;
    std::string input_names;
    std::string in_quant;
    std::string out_quant;

    NNTR_THROW_IF(graphs_info.find(graph_name) == graphs_info.end(),
                  std::runtime_error)
        << graph_name << " does not exist in model binary config"
        << binary_config_path << "!";

    auto &current_graphs_info = graphs_info[graph_name];
    std::vector<ml::train::TensorDim::IO_TensorType> model_inputs;

    for (const auto &[tensor_name, tensor_object] :
         current_graphs_info.raw_inputs) {
      if (uses_embedding && tensor_name == "inputs_embeds") {
        auto input_shape = tensor_object.dimensions;
        int input_size = input_shape[0];
        std::string input_shape_string = std::to_string(input_shape[0]);
        for (int i = 1; i < input_shape.size() - 1; i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string(input_shape[i]);
          input_size *= input_shape[i];
        }
        current_model->addLayer(createLayer(
            "embedding",
            {withKey("name", tensor_name), withKey("in_dim", vocab_size),
             withKey("input_shape", input_shape_string),
             withKey("out_dim", input_shape.back())}));

        model_inputs.push_back((float *)allocate(sizeof(float) * input_size));
      } else {
        auto input_shape = tensor_object.dimensions;
        std::string input_shape_string = std::to_string(input_shape[0]);
        for (int i = 1; i < input_shape.size(); i++) {
          input_shape_string += ":";
          input_shape_string += std::to_string(input_shape[i]);
        }
        current_model->addLayer(createLayer(
            "input", {withKey("name", tensor_name),
                      //  withKey("input_dtype",
                      //  qnn_to_nntrainer_datatype(tensor_object.data_type)),
                      withKey("input_shape", input_shape_string)}));
        model_inputs.push_back(get_qnn_input_data(tensor_object));
      }

      if (!input_names.empty()) {
        input_names += ", ";
      }
      input_names += tensor_name;

      if (!in_quant.empty()) {
        in_quant += ",";
      }
      in_quant += tensor_name;
      in_quant += ":";
      in_quant += std::to_string(tensor_object.scale);
      in_quant += ":";
      in_quant += std::to_string(tensor_object.offset);
    }

    for (const auto &[tensor_name, tensor_object] :
         current_graphs_info.raw_outputs) {
      if (!out_dim.empty()) {
        out_dim += ",";
      }
      out_dim += std::to_string(tensor_object.dimensions[0]);
      for (int i = 1; i < tensor_object.dimensions.size(); i++) {
        out_dim += ":";
        out_dim += std::to_string(tensor_object.dimensions[i]);
      }

      if (!out_data_format.empty()) {
        out_data_format += ",";
      }
      out_data_format += qnn_to_nntrainer_datatype(tensor_object.data_type);

      if (!out_tensor_format.empty()) {
        out_tensor_format += ",";
      }
      out_tensor_format += "OUT_TENSOR";

      if (!out_quant.empty()) {
        out_quant += ",";
      }
      out_quant += tensor_name;
      out_quant += ":";
      out_quant += std::to_string(tensor_object.scale);
      out_quant += ":";
      out_quant += std::to_string(tensor_object.offset);
    }

    LayerHandle qnn_layer = createLayer(
        "qnn_graph",
        {withKey("name", graph_name), withKey("path", model_file_name),
         withKey("dim", out_dim), withKey("tensor_dtype", out_data_format),
         withKey("tensor_type", out_tensor_format),
         withKey("input_layers", input_names),
         withKey("input_quant_param", in_quant),
         withKey("output_quant_param", out_quant), withKey("engine", "qnn")});
    current_model->addLayer(qnn_layer);

    current_model->setProperty({withKey("batch_size", 1), withKey("epochs", 1),
                                withKey("model_tensor_type", "UINT16-UINT16")});

    auto optimizer = createOptimizer("sgd", {withKey("learning_rate", 0.001)});
    current_model->setOptimizer(std::move(optimizer));

    status = current_model->compile(ExecutionMode::INFERENCE);
    if (status) {
      throw std::invalid_argument("Model compilation failed!");
    }

    status = current_model->initialize(ExecutionMode::INFERENCE);
    if (status) {
      throw std::invalid_argument("Model initialization failed!");
    }

    models[graph_name] = {current_graphs_info, std::move(current_model),
                          model_inputs};
  }
}

void causallm::Quick_Dot_AI_QNN::load_weight(const std::string &weight_path) {
  for (const auto &[key, value] : models) {
    value.model_handle->load(model_file_name, ModelFormat::MODEL_FORMAT_QNN);
  }
  // Allocate tensors for inference - required for input/output buffers
  for (auto &[key, value] : models) {
    value.model_handle->allocate(ExecutionMode::INFERENCE);
  }
}

void causallm::Quick_Dot_AI_QNN::save_weight(const std::string &weight_path) {
  // Unimplemented.
}

void causallm::Quick_Dot_AI_QNN::setupParameters(json &cfg,
                                                 json &generation_cfg,
                                                 json &nntr_cfg) {
  // Read nntr_config parameters
  LOGD("----------------in Quick_Dot_AI_QNN : setupParameters");  
  model_file_name = nntr_cfg["model_file_name"].get<std::string>();
  LOGD("----------------binary_config_path : %s", model_file_name.c_str());    
  binary_config_path = nntr_cfg["binary_config_path"].get<std::string>();
  LOGD("----------------binary_config_path : %s", binary_config_path.c_str());
  graphs_to_use = nntr_cfg["graphs_to_use"].get<std::vector<std::string>>();
  for(auto s : graphs_to_use){
    LOGD("----------------graphs_to_use : %s", s.c_str());
  }
  vocab_size = cfg["vocab_size"].get<int>();
  LOGD("----------------vocab size : %d", vocab_size);
}

void causallm::Quick_Dot_AI_QNN::constructModel() {
  // Unimplemented.
}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createTransformerDecoderBlock(
    const int layer_id, std::string input_name) {
  // Unimplemented.
  return std::vector<LayerHandle>();
}

std::vector<LayerHandle> causallm::Quick_Dot_AI_QNN::createAttention(
    const int layer_id, int seq_len, int n_heads, int head_dim,
    std::string query_name, std::string key_name, std::string value_name) {
  // Unimplemented.
  return std::vector<LayerHandle>();
}

std::vector<LayerHandle>
causallm::Quick_Dot_AI_QNN::createMlp(const int layer_id, int dim,
                                      int hidden_dim, std::string input_name) {
  // Unimplemented.
  return std::vector<LayerHandle>();
}

void causallm::Quick_Dot_AI_QNN::registerCustomLayers() {
  // Unimplemented.
}
