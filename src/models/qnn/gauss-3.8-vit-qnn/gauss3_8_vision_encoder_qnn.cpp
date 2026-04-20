// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3_8_vision_encoder_qnn.cpp
 * @brief  QNN model implementation for vision encoder model only.
 * @note   This class is not to be executed alone.
 *
 */

#include "gauss3_8_vision_encoder_qnn.h"
#include "factory.h"
#include "nntrainer_error.h"

/**
 * @brief Auto-registration via constructor attribute
 *
 * This function runs automatically when the shared library is loaded
 * (before main()). It registers all custom models with the CausalLM Factory.
 *
 */
__attribute__((constructor)) static void register_custom_models() {
  causallm::Factory::Instance().registerModel(
      "Gauss_3_8_VEncoder_QNN",
      [](causallm::json cfg, causallm::json generation_cfg,
         causallm::json nntr_cfg) {
        return std::make_unique<causallm::Gauss3_8_Vision_Encoder_QNN>(
            cfg, generation_cfg, nntr_cfg);
      });
}

causallm::Gauss3_8_Vision_Encoder_QNN::~Gauss3_8_Vision_Encoder_QNN() {
  if (image_newline_mmap_ptr != nullptr &&
      image_newline_mmap_ptr != MAP_FAILED) {
    ::munmap(image_newline_mmap_ptr, image_newline_mmap_size);
    image_newline_mmap_ptr = nullptr;
    image_newline_mmap_size = 0;
  }
}

TensorInfo causallm::Gauss3_8_Vision_Encoder_QNN::get_input_info() {
  std::string graph_name = graphs_to_use[0];
  LOGD("graph name : %s", graph_name.c_str());
  auto &[model_info, model, model_input] = models[graph_name];
  return model_info.raw_inputs[0].second;
}

TensorInfo causallm::Gauss3_8_Vision_Encoder_QNN::get_output_info() {
  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];
  return model_info.raw_outputs[0].second;
}

void causallm::Gauss3_8_Vision_Encoder_QNN::run(const WSTR prompt,
                                                bool do_sample,
                                                const WSTR system_prompt,
                                                const WSTR tail_prompt,
                                                bool log_output) {
  // Unimplemented
}

// Function to select the best resolution
std::pair<int, int> select_best_resolution(
    const std::pair<int, int> &original_size, // (height, width)
    const std::vector<std::pair<int, int>>
        &possible_resolutions // list of (height, width)
) {
  int original_height = original_size.first;
  int original_width = original_size.second;

  std::pair<int, int> best_fit = {0, 0};
  long long max_effective_resolution = 0;
  long long min_wasted_resolution = std::numeric_limits<long long>::max();

  for (const auto &res : possible_resolutions) {
    int height = res.first;
    int width = res.second;

    double scale = std::min(static_cast<double>(width) / original_width,
                            static_cast<double>(height) / original_height);

    int downscaled_width = static_cast<int>(original_width * scale);
    int downscaled_height = static_cast<int>(original_height * scale);

    long long effective_resolution =
        std::min(static_cast<long long>(downscaled_width) * downscaled_height,
                 static_cast<long long>(original_width) * original_height);

    long long wasted_resolution =
        static_cast<long long>(width) * height - effective_resolution;

    if (effective_resolution > max_effective_resolution ||
        (effective_resolution == max_effective_resolution &&
         wasted_resolution < min_wasted_resolution)) {
      max_effective_resolution = effective_resolution;
      min_wasted_resolution = wasted_resolution;
      best_fit = {height, width};
    }
  }

  return best_fit;
}

void causallm::Gauss3_8_Vision_Encoder_QNN::requantEmbedding(void *from,
                                                             void *to,
                                                             size_t length) {
  auto output_info = get_output_info();
  auto input_info = get_input_info();
  std::string encoderOutputDataType = output_info.data_type;
  std::string modelInputDataType = input_info.data_type;
  double requant_scale = output_info.scale / input_info.scale;
  double requant_offset =
      requant_scale * output_info.offset - input_info.offset;


  LOGD ("%d : %s, %s, %f, %f", length, encoderOutputDataType.c_str (),
      modelInputDataType.c_str (), requant_scale, requant_offset);

  for (int i = 0; i < length; i++) {
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
    }
    if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
        modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t *>(to)[i] = static_cast<uint8_t>(
          requant_scale * static_cast<uint16_t *>(from)[i] + requant_offset);
    }
    if (encoderOutputDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
        modelInputDataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t *>(to)[i] = static_cast<uint16_t>(
          requant_scale * static_cast<uint16_t *>(from)[i] + requant_offset);
    }
  }
}

causallm::multimodal_pointer causallm::Gauss3_8_Vision_Encoder_QNN::run_image(
    const WSTR prompt, multimodal_pointer image, int image_height,
    int image_width, bool do_sample, const WSTR system_prompt,
    const WSTR tail_prompt, bool log_output) {

  std::string graph_name = graphs_to_use[0];
  auto &[model_info, model, model_input] = models[graph_name];

  auto input_info = get_input_info();
  auto output_info = get_output_info();

  int num_bytes_per_inference = GraphParser::get_tensor_size(input_info);
  
  NNTR_THROW_IF(image.second % num_bytes_per_inference, std::invalid_argument)
      << "Image input data size " << image.second << " is not a multiple of "
      << num_bytes_per_inference;

  int num_inference = image.second / (num_bytes_per_inference*(sizeof(float)/sizeof(uint16_t)));

  LOGD("image.second %d, num_bytes_per_inference : %d, num_inference: %d", image.second, num_bytes_per_inference, num_inference);

  auto [height, width] = select_best_resolution(
      std::make_pair(image_height, image_width), grid_pinpoints);
  height /= image_patch_size;
  width /= image_patch_size;

  int num_tokens =
      (1 + height * width) * encode_patch_size * encode_patch_size +
      height * encode_patch_size;

  int out_embedding_size = output_info.dimensions[2];
  int base_image_offset = encode_patch_size * encode_patch_size;
  int next_row_offset = width * encode_patch_size + 1;
  int next_patch_offset = (width * encode_patch_size + 1) * encode_patch_size;
  int output_bw = GraphParser::get_tensor_bit_width (output_info);
  int total_embedding_size = num_tokens * out_embedding_size * output_bw;

  LOGD ("[REQUANT-DEBUG] num_tokens=%d out_embedding_size=%d output_bw=%d "
        "total_embedding_size=%d num_inference=%d h=%d w=%d encode_patch_size=%d "
        "image_newline.second=%zu",
      num_tokens, out_embedding_size, output_bw, total_embedding_size,
      num_inference, height, width, encode_patch_size, image_newline.second);

  void *my_output = malloc (total_embedding_size);

  NNTR_THROW_IF(out_embedding_size * output_bw != image_newline.second,
                std::invalid_argument)
      << "Image_newline token tensor size expected to be "
      << out_embedding_size * output_bw << ", but is actually "
      << image_newline.second;

  for (int i = 0; i < num_inference; i++) {
    std::string qnn_dtype_input = input_info.data_type;
    if (qnn_dtype_input == "QNN_DATATYPE_UFIXED_POINT_16") {
      memcpy(std::get<uint16_t *>(model_input[0]), image.first,
             num_bytes_per_inference);
    } else if (qnn_dtype_input == "QNN_DATATYPE_UFIXED_POINT_8") {
      memcpy(std::get<uint8_t *>(model_input[0]), image.first,
             num_bytes_per_inference);
    } else {
      throw std::invalid_argument("qnn_dtype_input is " + qnn_dtype_input);
    }
    auto qnn_output = model->inference(1, model_input)[0];
    void *vision_encoder_output = std::visit(
        [](auto *p) -> void * { return static_cast<void *>(p); }, qnn_output);

    int vision_encoder_output_size = GraphParser::get_tensor_size(output_info);

    NNTR_THROW_IF(encode_patch_size * encode_patch_size * out_embedding_size *
                          output_bw !=
                      vision_encoder_output_size,
                  std::invalid_argument)
        << "Patch output length expected to be "
        << encode_patch_size * encode_patch_size * out_embedding_size *
               output_bw
        << ", but is actually " << vision_encoder_output_size;

    if (i == 0) {
      // Base image
      requantEmbedding(vision_encoder_output, my_output,
                       encode_patch_size * encode_patch_size *
                           out_embedding_size);
    } else {
      auto patch_index = std::pair((i - 1) / width, (i - 1) % width);
      auto patch_offset = patch_index.first * next_patch_offset +
                          patch_index.second * encode_patch_size;
      for (size_t h = 0; h < encode_patch_size; h++) {
        requantEmbedding(
            (uint8_t *)vision_encoder_output +
                h * encode_patch_size * out_embedding_size * output_bw,
            (uint8_t *)my_output +
                (base_image_offset + patch_offset + h * next_row_offset) *
                    out_embedding_size * output_bw,
            encode_patch_size * out_embedding_size);
        if (patch_index.second == (width - 1)) {
          memcpy((uint8_t *)my_output +
                     (base_image_offset + patch_offset + h * next_row_offset +
                      encode_patch_size) *
                         out_embedding_size * output_bw,
                 image_newline.first, out_embedding_size * output_bw);
        }
      }
    }
  }
  std::cout << "run_image done!" << std::endl;
  return std::make_pair(my_output,
                        num_tokens * out_embedding_size *
                            GraphParser::get_tensor_bit_width(output_info));
}
