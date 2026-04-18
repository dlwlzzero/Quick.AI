// SPDX-License-Identifier: Apache-2.0
/**
 * @file   gauss3.8_vision_encoder_qnn.h
 * @brief  QNN model extension template
 * @note   This file demonstrates how to create a custom QNN Quick.AI model.
 *
 */

#ifndef __GAUSS_3_8_VISION_ENCODER_QNN_H__
#define __GAUSS_3_8_VISION_ENCODER_QNN_H__

#include "nntrainer_error.h"
#include "quick_dot_ai_qnn.h"
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace causallm {

/**
 * @brief Gauss3_8_Vision_Encoder_QNN class
 * @note  This class runs vision encoder QNN class and that's all.
 *
 */
class Gauss3_8_Vision_Encoder_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "Gauss_3_8_Visual_QNN";

  Gauss3_8_Vision_Encoder_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {
    // Load image_newline file using mmap
    if (nntr_cfg.contains("image_newline_path")) {
      std::string image_newline_path = nntr_cfg["image_newline_path"];

      int fd = ::open(image_newline_path.c_str(), O_RDONLY);
      NNTR_THROW_IF((fd == -1), std::invalid_argument)
          << "Cannot open file: " << image_newline_path;

      struct stat st {};
      NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
          << "Cannot get file info (fstat): " << image_newline_path;

      image_newline_mmap_size = static_cast<size_t>(st.st_size);
      image_newline_mmap_ptr = ::mmap(nullptr, image_newline_mmap_size,
                                      PROT_READ, MAP_PRIVATE, fd, 0);
      ::close(fd);

      NNTR_THROW_IF((image_newline_mmap_ptr == MAP_FAILED), std::runtime_error)
          << "mmap failed for: " << image_newline_path;

      (void)::posix_madvise(image_newline_mmap_ptr, image_newline_mmap_size,
                            POSIX_MADV_RANDOM);

      image_newline = {image_newline_mmap_ptr, image_newline_mmap_size};
    }
  }

  ~Gauss3_8_Vision_Encoder_QNN();

  TensorInfo get_input_info();
  TensorInfo get_output_info();

  void run(const WSTR prompt, bool do_sample = false,
           const WSTR system_prompt = "", const WSTR tail_prompt = "",
           bool log_output = true) override;

  multimodal_pointer run_image(const WSTR prompt, multimodal_pointer image,
                               int image_height, int image_width,
                               bool do_sample = false,
                               const WSTR system_prompt = "",
                               const WSTR tail_prompt = "",
                               bool log_output = true);

private:
  // Also hard-coded in Goka's genie
  std::vector<std::pair<int, int>> grid_pinpoints = {
      {512, 1024},  {512, 1536},  {512, 2048},  {512, 2560},  {512, 3072},
      {512, 3584},  {512, 4096},  {512, 4608},  {512, 5120},  {512, 5632},
      {512, 6144},  {1024, 512},  {1024, 1024}, {1024, 1536}, {1024, 2048},
      {1024, 2560}, {1024, 3072}, {1536, 512},  {1536, 1024}, {1536, 1536},
      {1536, 2048}, {2048, 512},  {2048, 1024}, {2048, 1536}, {2560, 512},
      {2560, 1024}, {3072, 512},  {3072, 1024}, {3584, 512},  {4096, 512},
      {4608, 512},  {5120, 512},  {5632, 512},  {6144, 512}};
  int image_patch_size = 512;
  int encode_patch_size = 16;

  void requantEmbedding(void *from, void *to, size_t length);
  multimodal_pointer image_newline;

  // mmap state for image_newline (for cleanup in destructor)
  void *image_newline_mmap_ptr = nullptr;
  size_t image_newline_mmap_size = 0;
};

} // namespace causallm

#endif /* __GAUSS_3_8_VISION_ENCODER_QNN_H__ */