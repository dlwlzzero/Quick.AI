#ifndef __QNN_KV_CACHE_MANAGER_H__
#define __QNN_KV_CACHE_MANAGER_H__

#include "generate_qnn_utils.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace causallm {

class QnnKvCacheManager {
public:
  void clear();

  void addLayerRowLength(int row_length);
  int addGenerationCache(const std::string &name, uint8_t *data, int byte_size,
                         int row_length, bool is_key);
  void addPrefillCache(uint8_t *data, int byte_size, int row_length,
                       int generation_index, bool is_key);

  const std::unordered_map<std::string, int> &generationIndexByName() const {
    return generation_kv_index_by_name_;
  }

  void setPrefillOutputBindings(std::vector<QnnKvOutputBinding> bindings);
  void setGenerationOutputBindings(std::vector<QnnKvOutputBinding> bindings);

  size_t generationCacheCount() const { return generation_caches_.size(); }
  size_t prefillCacheCount() const { return prefill_caches_.size(); }
  size_t prefillOutputBindingCount() const {
    return prefill_output_kv_bindings_.size();
  }
  size_t generationOutputBindingCount() const {
    return generation_output_kv_bindings_.size();
  }

  int length() const { return kv_len_; }
  void setLength(int length);
  void advance(int delta);

  void reset();
  void resetPrefillInputs();
  void syncGenerationToPrefill();

  void appendPrefillOutputs(const std::vector<IO_TensorType> &step_outputs,
                            int target_position, int rows, int src_row_length,
                            const std::string &graph_name);
  void appendGenerationOutputs(const std::vector<IO_TensorType> &step_outputs,
                               int target_position, int rows,
                               int src_row_length,
                               const std::string &graph_name);

  void save(const std::string &path, const std::string &architecture) const;
  void load(const std::string &path, const std::string &architecture,
            int max_length);

private:
  struct GenerationCache {
    std::string name;
    uint8_t *data = nullptr;
    int byte_size = 0;
    int row_length = 0;
    bool is_key = false;
  };

  struct PrefillCache {
    uint8_t *data = nullptr;
    int byte_size = 0;
    int row_length = 0;
    int generation_index = -1;
    bool is_key = false;
  };

  std::vector<GenerationCache> generation_caches_;
  std::vector<PrefillCache> prefill_caches_;
  std::vector<int> layer_row_lengths_;
  std::vector<QnnKvOutputBinding> prefill_output_kv_bindings_;
  std::vector<QnnKvOutputBinding> generation_output_kv_bindings_;
  std::unordered_map<std::string, int> generation_kv_index_by_name_;
  int kv_len_ = 0;
};

} // namespace causallm

#endif // __QNN_KV_CACHE_MANAGER_H__
