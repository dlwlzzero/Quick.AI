#ifndef GRAPH_PARSER_H
#define GRAPH_PARSER_H

#include "../../../nntrainer/Applications/CausalLM/json.hpp"
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

struct TensorInfo {
  std::string name;
  std::vector<int> dimensions;
  std::string data_type;
  double scale;
  int offset;
};

struct GraphInfo {
  std::string graph_name;
  std::vector<std::pair<std::string, TensorInfo>> raw_inputs;
  std::vector<std::pair<std::string, TensorInfo>> raw_outputs;
};

class GraphParser {
public:
  GraphParser();
  ~GraphParser();

  std::map<std::string, GraphInfo> parseJsonFile(const std::string &file_path);
  static int get_tensor_count(const TensorInfo &tensor_info);
  static int get_tensor_bit_width(const TensorInfo &tensor_info);
  static int get_tensor_size(const TensorInfo &tensor_info);

private:
  GraphInfo extractGraphInfo(const json &graph_object);
  TensorInfo extractTensorInfo(const json &tensor_object);
};

#endif // GRAPH_PARSER_H