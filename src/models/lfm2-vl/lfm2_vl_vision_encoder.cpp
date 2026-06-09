// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_vision_encoder.cpp
 * @brief  Quick.AI vision producer wrapping the nntrainer LFM2-VL SigLIP2 ViT
 *         + pixel-unshuffle + connector. Implements run_image() so the generic
 *         multimodal composer can pair it with the LFM2 LM consumer.
 */
#include "lfm2_vl_vision_encoder.h"

#include <factory.h>
#include <model_descriptor.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace causallm {

Lfm2VlVisionEncoder::Lfm2VlVisionEncoder(json &cfg, json &generation_cfg,
                                         json &nntr_cfg) {
  IMAGE_SIZE_ = cfg.value("image_size", 256u);
  PATCH_SIZE_ = cfg.value("patch_size", 16u);
  NUM_CHANNELS_ = cfg.value("num_channels", 3u);
  VIT_EMBED_ = cfg.value("hidden_size", 768u);
  DOWNSAMPLE_ = nntr_cfg.value("downsample_factor", 2u);

  json vit_nntr = nntr_cfg;
  vit_nntr["model_type"] = "embedding";
  vit_ =
    std::make_unique<Lfm2VlVisionTransformer>(cfg, generation_cfg, vit_nntr);

  unsigned int r = DOWNSAMPLE_;
  unsigned int connector_in = VIT_EMBED_ * r * r;
  unsigned int proj_hidden = nntr_cfg.value("projector_hidden_size", 2560u);
  unsigned int lm_hidden = nntr_cfg.value("lm_hidden_size", 1024u);
  connector_ =
    std::make_unique<Lfm2VlConnector>(connector_in, proj_hidden, lm_hidden);

  vision_weight_file_ = nntr_cfg.value("vision_model_file", "");
  connector_weight_file_ = nntr_cfg.value("connector_model_file", "");
}

void Lfm2VlVisionEncoder::initialize() { vit_->initialize(); }

void Lfm2VlVisionEncoder::load_weight(const std::string &weight_path) {
  if (!vision_weight_file_.empty())
    vit_->load_weight(vision_weight_file_);
  else
    vit_->load_weight(weight_path);

  if (connector_weight_file_.empty())
    throw std::runtime_error(
      "Lfm2VlVisionEncoder: nntr_config must set connector_model_file");
  connector_->loadWeights(connector_weight_file_);
}

multimodal_pointer Lfm2VlVisionEncoder::run_image(
  const WSTR /*prompt*/, multimodal_pointer image, int /*image_height*/,
  int /*image_width*/, bool /*do_sample*/, const WSTR /*system_prompt*/,
  const WSTR /*tail_prompt*/, bool /*log_output*/) {

  const size_t n_elems = expectedPixelElems();
  if (image.second < n_elems * sizeof(float))
    throw std::runtime_error("Lfm2VlVisionEncoder: pixel buffer too small");

  const char *tmp_dir = std::getenv("TMPDIR");
  std::string tmp_path =
    std::string(tmp_dir ? tmp_dir : "/data/local/tmp") + "/lfm2vl_pixels.bin";
  {
    std::ofstream of(tmp_path, std::ios::binary);
    of.write(reinterpret_cast<const char *>(image.first),
             static_cast<std::streamsize>(n_elems * sizeof(float)));
  }

  vit_->run(tmp_path, false, "", "", false);
  const std::vector<float> &feats = vit_->getLastFeatures();
  if (feats.empty())
    throw std::runtime_error("Lfm2VlVisionEncoder: ViT produced no features");

  unsigned int ph = IMAGE_SIZE_ / PATCH_SIZE_;
  unsigned int pw = IMAGE_SIZE_ / PATCH_SIZE_;
  unsigned int n_patches = ph * pw;

  auto unshuffled =
    pixelUnshuffle(feats, n_patches, VIT_EMBED_, ph, pw, DOWNSAMPLE_);
  unsigned int n_img_tokens = connector_->outTokens(n_patches);
  std::vector<float> embeds = connector_->forward(unshuffled, n_img_tokens);

  const size_t bytes = embeds.size() * sizeof(float);
  void *out = std::malloc(bytes);
  std::memcpy(out, embeds.data(), bytes);
  return {out, bytes};
}

__attribute__((constructor)) static void register_lfm2_vl_vision() {
  Factory::Instance().registerModel(
    "Lfm2VlVisionEncoder", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<Lfm2VlVisionEncoder>(cfg, generation_cfg,
                                                   nntr_cfg);
    });

  static const ModelDescriptor d = {"lfm2-vl-vision",
                                     "lfm2",
                                     "LFM2-VL Vision Encoder",
                                     QDA_RUNTIME_NATIVE,
                                     (1u << 0),
                                     QDA_CAP_VISION_ENCODER,
                                     "lfm2-vl-vision",
                                     "Lfm2VlVisionEncoder"};
  quick_dot_ai::register_model_descriptor(&d);
}

} // namespace causallm
