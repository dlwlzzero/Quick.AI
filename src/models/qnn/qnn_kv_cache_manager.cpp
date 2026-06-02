#include "qnn_kv_cache_manager.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <utility>

namespace causallm {
namespace {

constexpr char kKvCacheMagic[] = {'Q', 'A', 'I', 'Q', 'N', 'N', 'K', 'V'};
constexpr uint32_t kKvCacheVersion = 1;
constexpr uint8_t kKvCacheFill = 128;

void write_bytes(std::ofstream &out, const void *data, std::size_t size) {
  out.write(static_cast<const char *>(data), static_cast<std::streamsize>(size));
  if (!out) {
    throw std::runtime_error("Failed to write QNN KV cache");
  }
}

template <typename T> void write_value(std::ofstream &out, const T &value) {
  write_bytes(out, &value, sizeof(T));
}

void write_string(std::ofstream &out, const std::string &value) {
  const uint32_t size = static_cast<uint32_t>(value.size());
  write_value(out, size);
  if (size > 0) {
    write_bytes(out, value.data(), size);
  }
}

void read_bytes(std::ifstream &in, void *data, std::size_t size) {
  in.read(static_cast<char *>(data), static_cast<std::streamsize>(size));
  if (!in) {
    throw std::runtime_error("Failed to read QNN KV cache");
  }
}

template <typename T> T read_value(std::ifstream &in) {
  T value{};
  read_bytes(in, &value, sizeof(T));
  return value;
}

std::string read_string(std::ifstream &in) {
  const uint32_t size = read_value<uint32_t>(in);
  std::string value(size, '\0');
  if (size > 0) {
    read_bytes(in, &value[0], size);
  }
  return value;
}

} // namespace

void QnnKvCacheManager::clear() {
  generation_caches_.clear();
  prefill_caches_.clear();
  layer_row_lengths_.clear();
  prefill_output_kv_bindings_.clear();
  generation_output_kv_bindings_.clear();
  generation_kv_index_by_name_.clear();
  kv_len_ = 0;
}

void QnnKvCacheManager::addLayerRowLength(int row_length) {
  if (row_length <= 0) {
    throw std::runtime_error("Invalid QNN KV layer row length");
  }
  layer_row_lengths_.push_back(row_length);
}

int QnnKvCacheManager::addGenerationCache(const std::string &name,
                                          uint8_t *data, int byte_size,
                                          int row_length, bool is_key) {
  if (data == nullptr || byte_size <= 0 || row_length <= 0) {
    throw std::runtime_error("Invalid QNN generation KV cache tensor: " +
                             name);
  }

  const int index = static_cast<int>(generation_caches_.size());
  generation_caches_.push_back({name, data, byte_size, row_length, is_key});
  generation_kv_index_by_name_[name] = index;
  return index;
}

void QnnKvCacheManager::addPrefillCache(uint8_t *data, int byte_size,
                                        int row_length, int generation_index,
                                        bool is_key) {
  if (data == nullptr || byte_size <= 0 || row_length <= 0 ||
      generation_index < 0 ||
      generation_index >= static_cast<int>(generation_caches_.size())) {
    throw std::runtime_error("Invalid QNN prefill KV cache tensor");
  }

  prefill_caches_.push_back(
      {data, byte_size, row_length, generation_index, is_key});
}

void QnnKvCacheManager::setPrefillOutputBindings(
    std::vector<QnnKvOutputBinding> bindings) {
  prefill_output_kv_bindings_ = std::move(bindings);
}

void QnnKvCacheManager::setGenerationOutputBindings(
    std::vector<QnnKvOutputBinding> bindings) {
  generation_output_kv_bindings_ = std::move(bindings);
}

void QnnKvCacheManager::setLength(int length) {
  if (length < 0) {
    throw std::runtime_error("Invalid QNN KV cache length");
  }
  kv_len_ = length;
}

void QnnKvCacheManager::advance(int delta) {
  if (delta < 0 || kv_len_ + delta < 0) {
    throw std::runtime_error("Invalid QNN KV cache length delta");
  }
  kv_len_ += delta;
}

void QnnKvCacheManager::reset() {
  kv_len_ = 0;
  for (const auto &cache : generation_caches_) {
    std::memset(cache.data, kKvCacheFill, cache.byte_size);
  }
  resetPrefillInputs();
}

void QnnKvCacheManager::resetPrefillInputs() {
  for (const auto &cache : prefill_caches_) {
    std::fill_n(cache.data, cache.byte_size, kKvCacheFill);
  }
}

void QnnKvCacheManager::syncGenerationToPrefill() {
  resetPrefillInputs();

  if (kv_len_ <= 0) {
    return;
  }

#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(prefill_caches_.size()); i++) {
    const auto &prefill = prefill_caches_[i];
    const int generation_idx = prefill.generation_index;
    const int generation_layer_idx = generation_idx / 4;
    if (generation_idx < 0 ||
        generation_idx >= static_cast<int>(generation_caches_.size()) ||
        generation_layer_idx < 0 ||
        generation_layer_idx >= static_cast<int>(layer_row_lengths_.size())) {
      continue;
    }

    copy_kv_cache_window(prefill.data, prefill.row_length,
                         generation_caches_[generation_idx].data,
                         layer_row_lengths_[generation_layer_idx], kv_len_,
                         prefill.is_key);
  }
}

void QnnKvCacheManager::appendPrefillOutputs(
    const std::vector<IO_TensorType> &step_outputs, int target_position,
    int rows, int src_row_length, const std::string &graph_name) {
  std::vector<uint8_t *> kvs;
  kvs.reserve(generation_caches_.size());
  for (const auto &cache : generation_caches_) {
    kvs.push_back(cache.data);
  }

  append_outputs_to_kv_cache(step_outputs, prefill_output_kv_bindings_, kvs,
                             layer_row_lengths_, target_position, rows,
                             src_row_length, graph_name);
}

void QnnKvCacheManager::appendGenerationOutputs(
    const std::vector<IO_TensorType> &step_outputs, int target_position,
    int rows, int src_row_length, const std::string &graph_name) {
  std::vector<uint8_t *> kvs;
  kvs.reserve(generation_caches_.size());
  for (const auto &cache : generation_caches_) {
    kvs.push_back(cache.data);
  }

  append_outputs_to_kv_cache(step_outputs, generation_output_kv_bindings_, kvs,
                             layer_row_lengths_, target_position, rows,
                             src_row_length, graph_name);
}

void QnnKvCacheManager::save(const std::string &path,
                             const std::string &architecture) const {
  if (path.empty()) {
    throw std::runtime_error("QNN KV cache path is empty");
  }

  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out.is_open()) {
    throw std::runtime_error("Failed to open QNN KV cache for writing: " +
                             path);
  }

  write_bytes(out, kKvCacheMagic, sizeof(kKvCacheMagic));
  write_value(out, kKvCacheVersion);
  write_string(out, architecture);
  write_value(out, static_cast<int32_t>(kv_len_));

  write_value(out, static_cast<uint32_t>(layer_row_lengths_.size()));
  for (const int row_length : layer_row_lengths_) {
    write_value(out, static_cast<int32_t>(row_length));
  }

  write_value(out, static_cast<uint32_t>(generation_caches_.size()));
  for (const auto &cache : generation_caches_) {
    write_string(out, cache.name);
    write_value(out, static_cast<int32_t>(cache.byte_size));
    write_value(out, static_cast<int32_t>(cache.row_length));
    write_value(out, static_cast<uint8_t>(cache.is_key ? 1 : 0));
    write_bytes(out, cache.data, static_cast<std::size_t>(cache.byte_size));
  }
}

void QnnKvCacheManager::load(const std::string &path,
                             const std::string &architecture, int max_length) {
  if (path.empty()) {
    throw std::runtime_error("QNN KV cache path is empty");
  }

  std::ifstream in(path, std::ios::binary);
  if (!in.is_open()) {
    throw std::runtime_error("Failed to open QNN KV cache for reading: " +
                             path);
  }

  char magic[sizeof(kKvCacheMagic)]{};
  read_bytes(in, magic, sizeof(magic));
  if (std::memcmp(magic, kKvCacheMagic, sizeof(kKvCacheMagic)) != 0) {
    throw std::runtime_error("Invalid QNN KV cache file");
  }

  const uint32_t version = read_value<uint32_t>(in);
  if (version != kKvCacheVersion) {
    throw std::runtime_error("Unsupported QNN KV cache version");
  }

  const std::string saved_architecture = read_string(in);
  if (saved_architecture != architecture) {
    throw std::runtime_error("QNN KV cache architecture mismatch");
  }

  const int32_t saved_kv_len = read_value<int32_t>(in);
  if (saved_kv_len < 0 || saved_kv_len > max_length) {
    throw std::runtime_error("QNN KV cache length is out of range");
  }

  const uint32_t layer_count = read_value<uint32_t>(in);
  if (layer_count != layer_row_lengths_.size()) {
    throw std::runtime_error("QNN KV cache layer count mismatch");
  }
  for (uint32_t i = 0; i < layer_count; i++) {
    const int32_t row_length = read_value<int32_t>(in);
    if (row_length != layer_row_lengths_[i]) {
      throw std::runtime_error("QNN KV cache layer row length mismatch");
    }
  }

  const uint32_t tensor_count = read_value<uint32_t>(in);
  if (tensor_count != generation_caches_.size()) {
    throw std::runtime_error("QNN KV cache tensor count mismatch");
  }

  std::vector<std::vector<uint8_t>> loaded_tensors;
  loaded_tensors.reserve(tensor_count);
  for (uint32_t i = 0; i < tensor_count; i++) {
    const std::string name = read_string(in);
    const int32_t byte_size = read_value<int32_t>(in);
    const int32_t row_length = read_value<int32_t>(in);
    const uint8_t is_key = read_value<uint8_t>(in);

    const auto &cache = generation_caches_[i];
    if (name != cache.name || byte_size != cache.byte_size ||
        row_length != cache.row_length || (is_key != 0) != cache.is_key) {
      throw std::runtime_error("QNN KV cache tensor metadata mismatch");
    }

    std::vector<uint8_t> tensor(static_cast<std::size_t>(byte_size));
    read_bytes(in, tensor.data(), tensor.size());
    loaded_tensors.push_back(std::move(tensor));
  }

  for (size_t i = 0; i < loaded_tensors.size(); i++) {
    std::memcpy(generation_caches_[i].data, loaded_tensors[i].data(),
                loaded_tensors[i].size());
  }
  kv_len_ = saved_kv_len;
  resetPrefillInputs();
}

} // namespace causallm
