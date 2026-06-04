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

#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#define LOG_TAG "QuickAI"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#else
#define LOGD(fmt, ...) fprintf(stdout, fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) fprintf(stderr, fmt "\n", ##__VA_ARGS__)
#endif

namespace causallm {

/**
 * @brief Gauss3_8_Vision_Encoder_QNN class
 * @note  This class runs vision encoder QNN class and that's all.
 *
 */
class Gauss3_8_Vision_Encoder_QNN : public Quick_Dot_AI_QNN {

public:
  static constexpr const char *architectures = "Gauss_3_8_VEncoder_QNN";

  Gauss3_8_Vision_Encoder_QNN(json &cfg, json &generation_cfg, json &nntr_cfg)
      : Quick_Dot_AI_QNN(cfg, generation_cfg, nntr_cfg) {
    // Load image_newline file using mmap
    LOGD("--------------------------------- Vsion Ecoder QNN");

    if (nntr_cfg.contains("image_newline_path")) {
      std::string image_newline_path = nntr_cfg["image_newline_path"];
      LOGD("Vsion Ecoder QNN :  newline path %s", image_newline_path.c_str());
      int fd = ::open(image_newline_path.c_str(), O_RDONLY);
      NNTR_THROW_IF((fd == -1), std::invalid_argument)
          << "Cannot open file: " << image_newline_path;

      struct stat st {};
      NNTR_THROW_IF((::fstat(fd, &st) == -1), std::invalid_argument)
          << "Cannot get file info (fstat): " << image_newline_path;

      image_newline_mmap_size = static_cast<size_t>(st.st_size);
      image_newline_mmap_ptr =
          ::mmap(nullptr, image_newline_mmap_size, PROT_READ | PROT_WRITE,
                 MAP_PRIVATE, fd, 0);
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
                               bool log_output = true) override;
  void set_quant_param(float scale, int offset) override;

private:
  // Also hard-coded in Goka's genie
  std::vector<std::pair<int, int>> grid_pinpoints = {
      {512, 1024},  {512, 1536}, {512, 2048},  {512, 2560},  {512, 3072},
      {512, 3584},  {512, 4096}, {1024, 512},  {1024, 1024}, {1024, 1536},
      {1024, 2048}, {1536, 512}, {1536, 1024}, {2048, 512},  {2048, 1024},
      {2560, 512},  {3072, 512}, {3584, 512},  {4096, 512}};
  int image_patch_size = 512;
  int encode_patch_size = 16;

  void requantEmbedding(void *from, void *to, size_t length);
  multimodal_pointer image_newline;

  // mmap state for image_newline (for cleanup in destructor)
  void *image_newline_mmap_ptr = nullptr;
  size_t image_newline_mmap_size = 0;

  bool llm_quant_param_given = false;
  float llm_scale;
  int llm_offset;
};

} // namespace causallm

#endif /* __GAUSS_3_8_VISION_ENCODER_QNN_H__ */
