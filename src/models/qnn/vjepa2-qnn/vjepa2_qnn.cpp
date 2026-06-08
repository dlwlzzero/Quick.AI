// SPDX-License-Identifier: Apache-2.0
/**
 * @file   vjepa2_qnn.cpp
 * @brief  QNN model implementation for V-JEPA2 video encoder.
 * @note   This class is not to be executed alone.
 *
 */

#include "vjepa2_qnn.h"
#include "factory.h"
#include "nntrainer_error.h"
#include <model_descriptor.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

__attribute__((constructor)) static void register_vjepa2_qnn() {
  causallm::Factory::Instance().registerModel(
    "VJEPA2_QNN", [](causallm::json cfg, causallm::json generation_cfg,
                     causallm::json nntr_cfg) {
      return std::make_unique<causallm::VJEPA2_QNN>(cfg, generation_cfg,
                                                    nntr_cfg);
    });

  static const ModelDescriptor d = {"vjepa2-qnn",
                                    "vjepa",
                                    "V-JEPA 2 (QNN)",
                                    QDA_RUNTIME_NATIVE,
                                    (1u << 2),
                                    QDA_CAP_MULTIMODAL | QDA_CAP_MESSAGES_API |
                                      QDA_CAP_MULTI_IMAGE,
                                    "VJEPA2-QNN",
                                    "VJEPA2_QNN"};
  quick_dot_ai::register_model_descriptor(&d);
}

causallm::VJEPA2_QNN::~VJEPA2_QNN() {
  if (rotation_matrix_mmap_ptr_ != nullptr &&
      rotation_matrix_mmap_ptr_ != MAP_FAILED) {
    ::munmap(rotation_matrix_mmap_ptr_, rotation_matrix_mmap_size_);
    rotation_matrix_mmap_ptr_ = nullptr;
    rotation_matrix_mmap_size_ = 0;
  }
}

void causallm::VJEPA2_QNN::setupParameters(json &cfg, json &generation_cfg,
                                           json &nntr_cfg) {
  Quick_Dot_AI_QNN::setupParameters(cfg, generation_cfg, nntr_cfg);

  if (nntr_cfg.contains("rotation_matrix_path")) {
    rotation_matrix_path_ = nntr_cfg["rotation_matrix_path"].get<std::string>();
  }
}

TensorInfo causallm::VJEPA2_QNN::get_input_info() {
  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;
  (void)model_input;
  int idx =
    GraphParser::find_tensor_index(model_info.raw_inputs, "pixel_values_video");
  NNTR_THROW_IF(idx < 0, std::invalid_argument)
    << "pixel_values_video not found in graph inputs";
  return model_info.raw_inputs[idx];
}

TensorInfo causallm::VJEPA2_QNN::get_output_info() {
  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;
  (void)model_input;
  return model_info.raw_outputs[0];
}

void causallm::VJEPA2_QNN::loadRotationMatrix() {
  if (rotation_matrix_path_.empty())
    return;

  int fd = ::open(rotation_matrix_path_.c_str(), O_RDONLY);
  NNTR_THROW_IF(fd == -1, std::invalid_argument)
    << "Cannot open rotation_matrix file: " << rotation_matrix_path_;

  struct stat st {};
  NNTR_THROW_IF(::fstat(fd, &st) == -1, std::invalid_argument)
    << "Cannot fstat rotation_matrix file: " << rotation_matrix_path_;

  rotation_matrix_mmap_size_ = static_cast<size_t>(st.st_size);
  rotation_matrix_mmap_ptr_ =
    ::mmap(nullptr, rotation_matrix_mmap_size_, PROT_READ | PROT_WRITE,
           MAP_PRIVATE, fd, 0);
  ::close(fd);

  NNTR_THROW_IF(rotation_matrix_mmap_ptr_ == MAP_FAILED, std::runtime_error)
    << "mmap failed for rotation_matrix: " << rotation_matrix_path_;

  (void)::posix_madvise(rotation_matrix_mmap_ptr_, rotation_matrix_mmap_size_,
                        POSIX_MADV_RANDOM);

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;
  int idx =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rotation_matrix");
  NNTR_THROW_IF(idx < 0, std::invalid_argument)
    << "rotation_matrix not found in graph inputs";

  size_t expected_size =
    GraphParser::get_tensor_size(model_info.raw_inputs[idx]);
  NNTR_THROW_IF(rotation_matrix_mmap_size_ != expected_size,
                std::invalid_argument)
    << "rotation_matrix file size " << rotation_matrix_mmap_size_
    << " != expected " << expected_size;

  std::memcpy(std::get<uint8_t *>(model_input[idx]), rotation_matrix_mmap_ptr_,
              rotation_matrix_mmap_size_);
  LOGD("rotation_matrix loaded (%zu bytes)", rotation_matrix_mmap_size_);
}

void causallm::VJEPA2_QNN::initialize() {
  Quick_Dot_AI_QNN::initialize();

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  (void)model;

  // Cache input pointers by name
  int idx_rot =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rotation_matrix");
  int idx_pix =
    GraphParser::find_tensor_index(model_info.raw_inputs, "pixel_values_video");
  int idx_cos =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rope_cos");
  int idx_sin =
    GraphParser::find_tensor_index(model_info.raw_inputs, "rope_sin");

  NNTR_THROW_IF(idx_rot < 0, std::invalid_argument)
    << "rotation_matrix not found in graph inputs";
  NNTR_THROW_IF(idx_pix < 0, std::invalid_argument)
    << "pixel_values_video not found in graph inputs";
  NNTR_THROW_IF(idx_cos < 0, std::invalid_argument)
    << "rope_cos not found in graph inputs";
  NNTR_THROW_IF(idx_sin < 0, std::invalid_argument)
    << "rope_sin not found in graph inputs";

  rotation_matrix_input_ = std::get<uint8_t *>(model_input[idx_rot]);
  pixel_values_input_ = std::get<uint16_t *>(model_input[idx_pix]);
  rope_cos_input_ = std::get<uint16_t *>(model_input[idx_cos]);
  rope_sin_input_ = std::get<uint16_t *>(model_input[idx_sin]);

  // Load constant inputs
  loadRotationMatrix();

  // TODO: Compute rope_cos / rope_sin for V-JEPA2 vision-specific 3D RoPE.
  // Current placeholder: leave as zero (or pre-filled by QNN runtime if any).
  // Shape [1, 1, 3072, 64] each. Must quantize to UF16 using per-tensor
  // scale/offset once the exact formula is known.
  LOGD("VJEPA2_QNN: rope_cos/rope_sin computation TODO");
}

void causallm::VJEPA2_QNN::run(const WSTR prompt, bool do_sample,
                               const WSTR system_prompt, const WSTR tail_prompt,
                               bool log_output) {
  // Unimplemented — vision-only encoder
}

void causallm::VJEPA2_QNN::set_quant_param(float scale, int offset) {
  llm_quant_param_given_ = true;
  llm_scale_ = scale;
  llm_offset_ = offset;
}

void causallm::VJEPA2_QNN::requantEmbedding(void *from, void *to,
                                            size_t length) {
  auto output_info = get_output_info();
  std::string encoderOutputDataType = output_info.data_type;
  std::string modelInputDataType = "QNN_DATATYPE_UFIXED_POINT_16";
  NNTR_THROW_IF(!llm_quant_param_given_, std::runtime_error)
    << "Please give LLM quant param!";

  double requant_scale = output_info.scale / llm_scale_;
  double requant_offset = requant_scale * output_info.offset - llm_offset_;

  LOGD("%zu : %s, %s, %f, %f", length, encoderOutputDataType.c_str(),
       modelInputDataType.c_str(), requant_scale, requant_offset);

  for (size_t i = 0; i < length; i++) {
    if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
        modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t *>(to)[i] = static_cast<int8_t>(
        requant_scale * static_cast<int8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t *>(to)[i] = static_cast<int16_t>(
        requant_scale * static_cast<int8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t *>(to)[i] = static_cast<uint8_t>(
        requant_scale * static_cast<uint8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t *>(to)[i] = static_cast<uint16_t>(
        requant_scale * static_cast<uint8_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t *>(to)[i] = static_cast<int8_t>(
        requant_scale * static_cast<int16_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t *>(to)[i] = static_cast<int16_t>(
        requant_scale * static_cast<int16_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t *>(to)[i] = static_cast<uint8_t>(
        requant_scale * static_cast<uint16_t *>(from)[i] + requant_offset);
    } else if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t *>(to)[i] = static_cast<uint16_t>(
        requant_scale * static_cast<uint16_t *>(from)[i] + requant_offset);
    }
  }
}

causallm::multimodal_pointer
causallm::VJEPA2_QNN::run_image(const WSTR prompt, multimodal_pointer image,
                                int image_height, int image_width,
                                bool do_sample, const WSTR system_prompt,
                                const WSTR tail_prompt, bool log_output) {

  auto start_total = std::chrono::high_resolution_clock::now();

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];

  auto input_info = get_input_info();
  auto output_info = get_output_info();

  int num_bytes_per_inference = GraphParser::get_tensor_size(input_info);
  int num_elements_per_inference = GraphParser::get_tensor_count(input_info);

  NNTR_THROW_IF(image.second % (num_bytes_per_inference *
                                (sizeof(float) / sizeof(uint16_t))),
                std::invalid_argument)
    << "Video input data size " << image.second << " is not a multiple of "
    << num_bytes_per_inference * (sizeof(float) / sizeof(uint16_t));

  int num_inference = image.second / (num_bytes_per_inference *
                                      (sizeof(float) / sizeof(uint16_t)));

  LOGD("image.second %zu, num_bytes_per_inference : %d, num_inference: %d",
       image.second, num_bytes_per_inference, num_inference);

  int out_embedding_size = output_info.dimensions[2]; // 768
  int output_bw = GraphParser::get_tensor_bit_width(output_info);
  int num_tokens = output_info.dimensions[1]; // 3072
  int total_embedding_size = num_tokens * out_embedding_size * output_bw;

  void *my_output = malloc(total_embedding_size * num_inference);

  for (int i = 0; i < num_inference; i++) {
    auto src = ((float *)image.first) + i * num_elements_per_inference;
    quantize_uint16_memcpy(src, pixel_values_input_, num_elements_per_inference,
                           input_info.scale, input_info.offset);
    auto qnn_output = model->inference(1, model_input)[0];
    void *vision_encoder_output = std::visit(
      [](auto *p) -> void * { return static_cast<void *>(p); }, qnn_output);

    int vision_encoder_output_size = GraphParser::get_tensor_size(output_info);

    void *dest =
      static_cast<uint8_t *>(my_output) + i * vision_encoder_output_size;
    if (llm_quant_param_given_) {
      requantEmbedding(vision_encoder_output, dest,
                       vision_encoder_output_size / output_bw);
    } else {
      std::memcpy(dest, vision_encoder_output, vision_encoder_output_size);
    }
  }

  auto end_total = std::chrono::high_resolution_clock::now();
  auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    end_total - start_total)
                    .count();

  performance_metrics.prefill_tokens = static_cast<unsigned int>(num_tokens);
  performance_metrics.prefill_duration_ms = static_cast<double>(total_ms);
  performance_metrics.generation_tokens = 0;
  performance_metrics.generation_duration_ms = 0.0;
  performance_metrics.total_duration_ms = static_cast<double>(total_ms);
  performance_metrics.peak_memory_kb = getPeakMemoryKb();
  has_run_ = true;

  std::cout << "run_image done!" << std::endl;
  return std::make_pair(my_output, total_embedding_size * num_inference);
}
