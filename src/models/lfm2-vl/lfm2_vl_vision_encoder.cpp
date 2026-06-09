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
  // weight_path is the model_file_name already rebased onto the sub-model dir
  // by the multi-model loader. Use it directly for the ViT.
  vit_->load_weight(weight_path);

  if (connector_weight_file_.empty())
    throw std::runtime_error(
      "Lfm2VlVisionEncoder: nntr_config must set connector_model_file");
  // The loader only rebases model_file_name; resolve a relative connector path
  // against the same directory as the (rebased) ViT weight path.
  std::string conn = connector_weight_file_;
  if (!conn.empty() && conn[0] != '/') {
    const auto slash = weight_path.find_last_of('/');
    if (slash != std::string::npos)
      conn = weight_path.substr(0, slash + 1) + conn;
  }
  connector_->loadWeights(conn);
}

multimodal_pointer Lfm2VlVisionEncoder::run_image(
  const WSTR /*prompt*/, multimodal_pointer image, int /*image_height*/,
  int /*image_width*/, bool /*do_sample*/, const WSTR /*system_prompt*/,
  const WSTR /*tail_prompt*/, bool /*log_output*/) {

  const size_t n_elems = expectedPixelElems();
  if (image.first == nullptr || image.second < n_elems * sizeof(float))
    throw std::runtime_error("Lfm2VlVisionEncoder: pixel buffer too small");

  vit_->runFromPixels(reinterpret_cast<const float *>(image.first), n_elems,
                      false);
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
  if (out == nullptr)
    throw std::runtime_error("Lfm2VlVisionEncoder: malloc failed");
  std::memcpy(out, embeds.data(), bytes);
  return {out, bytes};
}

__attribute__((constructor)) static void register_lfm2_vl_vision() {
  // Register the Factory creator only. The vision encoder is an INTERNAL
  // sub-model of the lfm2-vl composite (loaded via the multi-model config's
  // Factory::create), not a user-facing model. We intentionally do NOT register
  // a catalog ModelDescriptor for it: ModelCatalog.resolve(family,rt,backend)
  // matches the first descriptor in all() (not just selectable ones), so a
  // descriptor with family "lfm2" would shadow the real "lfm2-vl" composite in
  // the picker. Factory registration alone is sufficient for the composite.
  Factory::Instance().registerModel(
    "Lfm2VlVisionEncoder", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<Lfm2VlVisionEncoder>(cfg, generation_cfg,
                                                   nntr_cfg);
    });
}

} // namespace causallm
