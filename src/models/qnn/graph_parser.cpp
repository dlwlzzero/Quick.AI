#include "graph_parser.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include "nntrainer_error.h"

GraphParser::GraphParser() {}

GraphParser::~GraphParser() {}

std::map<std::string, GraphInfo>
GraphParser::parseJsonFile(const std::string &file_path) {
  std::map<std::string, GraphInfo> graphs_info;
  try {
    std::ifstream file(file_path);
    NNTR_THROW_IF(!file.is_open(), std::invalid_argument)
        << "Failed to open file: " << file_path;

    auto json_data_ = json::parse(file);
    file.close();

    // Parse graphs info
    auto &graphs = json_data_["info"]["graphs"];
    for (const auto &graph_object : graphs) {
      GraphInfo graph_info = extractGraphInfo(graph_object);
      graphs_info[graph_info.graph_name] = graph_info;
    }

    return graphs_info;
  } catch (const std::exception &e) {
    std::cerr << "Error parsing JSON file: " << e.what() << std::endl;
    return graphs_info;
  }
}

GraphInfo GraphParser::extractGraphInfo(const json &graph_object) {
  GraphInfo graph_info;

  auto graph_info_json = graph_object["info"];
  graph_info.graph_name = graph_info_json["graphName"];

  auto graph_inputs = graph_info_json["graphInputs"];
  auto graph_outputs = graph_info_json["graphOutputs"];

  for (const auto &element : graph_inputs) {
    auto tensor_info = extractTensorInfo(element);
    graph_info.raw_inputs.push_back({tensor_info.name, tensor_info});
  }

  for (const auto &element : graph_outputs) {
    auto tensor_info = extractTensorInfo(element);
    graph_info.raw_outputs.push_back({tensor_info.name, tensor_info});
  }

  return graph_info;
}

TensorInfo GraphParser::extractTensorInfo(const json &tensor_object) {
  TensorInfo tensor_info;

  auto tensor_info_json = tensor_object["info"];

  tensor_info.name = tensor_info_json["name"];
  tensor_info.dimensions =
      tensor_info_json["dimensions"].get<std::vector<int>>();
  tensor_info.data_type = tensor_info_json["dataType"];
  tensor_info.scale =
      tensor_info_json["quantizeParams"]["scaleOffset"]["scale"];
  tensor_info.offset =
      tensor_info_json["quantizeParams"]["scaleOffset"]["offset"];

  return tensor_info;
}

int GraphParser::get_tensor_count(const TensorInfo &tensor_info) {
  auto &tensor_shape = tensor_info.dimensions;

  int tensor_size = 1;
  for (const auto &tensor_dim : tensor_shape) {
    tensor_size *= tensor_dim;
  }
  return tensor_size;
}

int GraphParser::get_tensor_bit_width(const TensorInfo &tensor_info) {
  int bit_width;
  if (tensor_info.data_type == "QNN_DATATYPE_UFIXED_POINT_16") {
    bit_width = 2;
  } else if (tensor_info.data_type == "QNN_DATATYPE_UFIXED_POINT_8") {
    bit_width = 1;
  } else {
    throw std::invalid_argument("qnn tensor dtype is " + tensor_info.data_type);
  }
  return bit_width;
}

int GraphParser::get_tensor_size(const TensorInfo &tensor_info) {
  return get_tensor_bit_width(tensor_info) * get_tensor_count(tensor_info);
}
