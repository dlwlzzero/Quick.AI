// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_vision_encoder.h
 * @brief  Quick.AI vision producer wrapping the nntrainer LFM2-VL SigLIP2 ViT
 *         + pixel-unshuffle + connector. Implements the model-agnostic
 *         run_image() so the generic multimodal composer can pair it with the
 *         LFM2 LM consumer.
 */
#ifndef __LFM2_VL_VISION_ENCODER_H__
#define __LFM2_VL_VISION_ENCODER_H__

#include <transformer.h>

#include <lfm2_vl_connector.h>
#include <lfm2_vl_vision_transformer.h>

#include <memory>
#include <string>

namespace causallm {

class Lfm2VlVisionEncoder : public Transformer {
public:
  static constexpr const char *architectures = "Lfm2VlVisionEncoder";

  Lfm2VlVisionEncoder(json &cfg, json &generation_cfg, json &nntr_cfg);
  ~Lfm2VlVisionEncoder() override = default;

  void initialize() override;
  void initialize(const std::string &native_lib_dir) override { initialize(); }
  void load_weight(const std::string &weight_path) override;

  multimodal_pointer run_image(const WSTR prompt, multimodal_pointer image,
                               int image_height, int image_width,
                               bool do_sample = false,
                               const WSTR system_prompt = "",
                               const WSTR tail_prompt = "",
                               bool log_output = true) override;

  void set_quant_param(float scale, int offset) override {
    out_scale_ = scale;
    out_offset_ = offset;
  }

  size_t expectedPixelElems() const override {
    return static_cast<size_t>(NUM_CHANNELS_) * IMAGE_SIZE_ * IMAGE_SIZE_;
  }

private:
  std::unique_ptr<Lfm2VlVisionTransformer> vit_;
  std::unique_ptr<Lfm2VlConnector> connector_;

  unsigned int IMAGE_SIZE_ = 256;
  unsigned int PATCH_SIZE_ = 16;
  unsigned int NUM_CHANNELS_ = 3;
  unsigned int VIT_EMBED_ = 768;
  unsigned int DOWNSAMPLE_ = 2;
  std::string connector_weight_file_;
  std::string vision_weight_file_;

  float out_scale_ = 1.0f;
  int out_offset_ = 0;
};

} // namespace causallm

#endif // __LFM2_VL_VISION_ENCODER_H__
