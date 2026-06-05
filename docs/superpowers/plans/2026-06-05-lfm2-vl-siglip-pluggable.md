# LFM2-VL(SigLIP + LFM2) pluggable composer 통합 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** nntrainer의 모놀리식 LFM2-VL-450M을 vision 인코더 + LFM2 LM 두 개의 독립 로드 가능한 모델로 분해하여, Quick.AI의 기존 generic `execute_multimodal` composer(CPU 개방)로 페어링하고, 앱에서 vision+LLM 믹스앤매치로 골라 이미지+텍스트 추론을 한다.

**Architecture:** nntrainer에 `Lfm2VlVisionEncoder`(ViT+connector를 묶어 `run_image`가 LM 임베딩 공간 FP32 1024-dim 출력) 래퍼를 신규 추가하고, `Lfm2CausalLM`에 composer가 요구하는 base 가상(`embeddingBytesPerToken`/`lookupEmbedding(int)const`/`get_embedding_info`)을 오버라이드한다. Quick.AI api는 두 모델을 Factory 등록 + 공개 descriptor 추가하고, `#ifdef ENABLE_QNN` 멀티모달 경로를 CPU로 개방하며 이미지 마커를 LFM2(`<image>`=396)에 맞춘다. 앱은 `loadMultimodalHandleByName` JNI 노출 + 믹스앤매치 picker + SigLIP-NaFlex 전처리(MVP 고정 256²)를 추가한다.

**Tech Stack:** C++17 (nntrainer CausalLM, `causallm::Factory`/`Transformer`), Quick.AI C API(`libquick_dot_ai_api.so`), Android NDK JNI(`quickai_jni.cpp`), Kotlin(Jetpack Compose, `QuickDotAI` AAR + `SampleTestAPP`). 검증: 헤드리스 `quick_dot_ai_test` → APK on-device.

**정답 오라클(oracle):** 기존 모놀리식 `Lfm2VlForConditionalGeneration`(`nntrainer/.../lfm2/lfm2-vl/lfm2_vl_model.cpp`)의 `run()` 출력. 모든 헤드리스 검증은 동일 이미지/프롬프트에서 이 경로의 생성 토큰열과 비교한다.

**빌드 환경:** 메모리 `quickai-android-build-env`(JDK/SDK 경로, `./build.sh --platform=android`, `apk_install_android.sh`, `ANDROID_SERIAL`) 및 `gauss-pluggable-bringup`(APK verify 절차) 참고.

---

## File Structure (변경/생성 파일 맵)

**nntrainer 서브모듈 (공개 코드):**
- Create `nntrainer/Applications/CausalLM/models/lfm2/lfm2-vl/lfm2_vl_vision_encoder.h` — `Lfm2VlVisionEncoder` 선언(ViT+connector 소유, `run_image` 오버라이드).
- Create `nntrainer/Applications/CausalLM/models/lfm2/lfm2-vl/lfm2_vl_vision_encoder.cpp` — 구현.
- Modify `nntrainer/Applications/CausalLM/models/lfm2/lfm2_causallm.h` — base 가상 3종 오버라이드 선언 + scratch 멤버.
- Modify `nntrainer/Applications/CausalLM/models/lfm2/lfm2_causallm.cpp` — 그 구현.
- Modify `nntrainer/Applications/CausalLM/models/lfm2/lfm2-vl/meson.build` (또는 상위 meson) — 새 .cpp 빌드 포함.
- Modify `nntrainer/Applications/CausalLM/main.cpp` — `Lfm2VlVisionEncoder` Factory 등록(헤드리스 단독 테스트용).

**Quick.AI api:**
- Modify `api/quick_dot_ai_api.cpp` — (a) `register_models()`에 `Lfm2ForCausalLM`+`Lfm2VlVisionEncoder` 등록, (b) 멀티모달 경로 CPU 개방, (c) 이미지 마커 LFM2 정합, (d) `run_vision_encoder` 픽셀 레이아웃 모델 주도화.
- Modify `api/model_descriptors_public.cpp` — `lfm2-450m`(LM) + `siglip2-vl-encoder`(VISION_ENCODER) descriptor 추가.

**Android:**
- Modify `Android/QuickDotAI/src/main/cpp/quickai_jni.cpp` — `loadMultimodalHandleByNameNative` 추가.
- Modify `Android/QuickDotAI/src/main/java/com/example/quickdotai/NativeCausalLm.kt` — external fun + 래퍼.
- Modify `Android/QuickDotAI/src/main/java/com/example/quickdotai/ModelCatalog.kt` — `visionEncoders()`/`pairableLlms()`.
- Create `Android/QuickDotAI/src/main/java/com/example/quickdotai/SigLipNaFlexImageProcessor.kt` — 256² 전처리.
- Modify `Android/SampleTestAPP/.../MainActivity.kt` — 믹스앤매치 picker UI + 페어 로드 + 전처리 분기.

**디바이스 에셋(빌드 산출물 아님, 수동 배치):**
- `/sdcard/Download/aistudio-mobile/models/siglip2-vl-encoder/` (ViT+connector 가중치 + nntr_config.json)
- `/sdcard/Download/aistudio-mobile/models/lfm2-450m/` (LFM2 LM 가중치 + tokenizer + 임베딩 bin + nntr_config.json)

---

## Milestone 1 — nntrainer: 모델 분해

### Task 1.1: `Lfm2CausalLM`에 composer용 base 가상 오버라이드 추가

**배경:** composer(`execute_multimodal`)는 base 가상 `embeddingBytesPerToken()`/`const void* lookupEmbedding(int)const`/`get_embedding_info()`를 호출한다. 현재 `Lfm2CausalLM`은 concrete `std::vector<float> lookupEmbedding(unsigned)`만 있어 base 가상은 기본값(0/nullptr)을 반환 → composer가 동작 불가.

**Files:**
- Modify: `nntrainer/Applications/CausalLM/models/lfm2/lfm2_causallm.h`
- Modify: `nntrainer/Applications/CausalLM/models/lfm2/lfm2_causallm.cpp`

- [ ] **Step 1: 헤더에 base 가상 오버라이드 + scratch 멤버 선언 추가**

`lfm2_causallm.h`의 `class Lfm2CausalLM` public 영역(기존 `std::vector<float> lookupEmbedding(unsigned int token_id);` 선언 아래)에 추가:

```cpp
  // ── Multimodal composer (base Transformer) interface ──────────────────
  // The generic api composer drives this LM through base-class virtuals.
  // These adapt the concrete FP32 embedding path to the model-agnostic API.

  /** Bytes of one token embedding (FP32 DIM). 0 until weights loaded. */
  size_t embeddingBytesPerToken() const override;

  /** Embedding of @p token_id as a raw FP32 row, or nullptr. Pointer is
   *  valid until the next call (per-call scratch buffer). */
  const void *lookupEmbedding(int token_id) const override;

  /** FP32 identity quant space: connector emits FP32 1024-dim directly. */
  std::pair<float, int> get_embedding_info() override { return {1.0f, 0}; }
```

그리고 private 영역(기존 `embedding_*` 캐시 멤버들 근처)에 추가:

```cpp
  /** Per-call scratch for the base-virtual lookupEmbedding(int) const. */
  mutable std::vector<float> emb_scratch_;
```

- [ ] **Step 2: cpp에 구현 추가**

`lfm2_causallm.cpp` 파일 끝(또는 기존 `lookupEmbedding` 정의 아래)에 추가. `DIM`은 base `Transformer`의 hidden-size 멤버(기존 concrete `lookupEmbedding`이 사용하는 것과 동일한 멤버명을 그 함수 본문에서 확인해 동일하게 사용한다 — 대개 `DIM`):

```cpp
size_t Lfm2CausalLM::embeddingBytesPerToken() const {
  // FP32 row of width DIM. Returns 0 if embedding weights not yet cached.
  if (!embedding_weight_cached_)
    return 0;
  return static_cast<size_t>(DIM) * sizeof(float);
}

const void *Lfm2CausalLM::lookupEmbedding(int token_id) const {
  if (!embedding_weight_cached_ || token_id < 0)
    return nullptr;
  // Reuse the concrete FP32 lookup, but it is non-const; cast away const for
  // the cache-only access (no logical state change). The result is copied
  // into the per-call scratch so the returned pointer stays valid until the
  // next call (the composer memcpy's it immediately).
  auto *self = const_cast<Lfm2CausalLM *>(this);
  emb_scratch_ = self->lookupEmbedding(static_cast<unsigned int>(token_id));
  if (emb_scratch_.empty())
    return nullptr;
  return emb_scratch_.data();
}
```

> 주의: concrete `lookupEmbedding(unsigned)`와 base `lookupEmbedding(int)const`는 시그니처가 달라 **오버로드**로 공존한다. cpp Step 2의 `self->lookupEmbedding(...)`는 `unsigned int` 인자라 concrete가 선택된다. 컴파일 시 모호성 경고가 나오면 concrete를 `lookupEmbeddingVec`로 개명하고 양쪽 호출부(이 함수 + `lfm2_vl_model.cpp`)를 맞춘다.

- [ ] **Step 3: 빌드 확인 (헤드리스 호스트 빌드)**

Run: `cd nntrainer && ./tools/package_android.sh 2>/dev/null; meson compile -C build 2>&1 | tail -20` *(실제 빌드 명령은 메모리 `quickai-android-build-env` 기준으로 대체)*
Expected: `Lfm2CausalLM` 컴파일 성공, 링크 에러 없음.

- [ ] **Step 4: Commit**

```bash
cd nntrainer && git add Applications/CausalLM/models/lfm2/lfm2_causallm.h Applications/CausalLM/models/lfm2/lfm2_causallm.cpp
git commit -m "feat(lfm2): expose base Transformer embedding virtuals for composer"
```

### Task 1.2: `Lfm2VlVisionEncoder` 래퍼 신규 작성 (ViT+connector → run_image)

**배경:** composer는 `models[0]`(vision)의 base 가상 `run_image()`를 호출해 LM 임베딩 공간의 임베딩을 얻는다. nntrainer엔 ViT(`Lfm2VlVisionTransformer`)와 connector(`Lfm2VlConnector`)가 분리돼 있고 오케스트레이터만 둘을 잇는다. 이를 하나의 Factory 모델로 감싼다. ViT의 `run()`은 파일 기반이므로 in-memory 픽셀을 임시 파일로 우회(오케스트레이터와 동일하게 검증된 경로).

**Files:**
- Create: `nntrainer/Applications/CausalLM/models/lfm2/lfm2-vl/lfm2_vl_vision_encoder.h`
- Create: `nntrainer/Applications/CausalLM/models/lfm2/lfm2-vl/lfm2_vl_vision_encoder.cpp`

- [ ] **Step 1: 헤더 작성**

`lfm2_vl_vision_encoder.h`:

```cpp
// SPDX-License-Identifier: Apache-2.0
/**
 * @file   lfm2_vl_vision_encoder.h
 * @brief  Loadable vision-encoder model for LFM2-VL: wraps the SigLIP2 ViT
 *         (Lfm2VlVisionTransformer) + pixel-unshuffle + Lfm2VlConnector so a
 *         single run_image() returns image embeddings already projected into
 *         the LFM2 LM embedding space (FP32, out_features wide).
 *         Registered with Factory under "Lfm2VlVisionEncoder".
 */
#ifndef __LFM2_VL_VISION_ENCODER_H__
#define __LFM2_VL_VISION_ENCODER_H__

#include <memory>
#include <transformer.h>

#include "lfm2_vl_connector.h"
#include "vision/lfm2_vl_vision_transformer.h"

namespace causallm {

class Lfm2VlVisionEncoder : public Transformer {
public:
  static constexpr const char *architectures = "Lfm2VlVisionEncoder";

  Lfm2VlVisionEncoder(json &cfg, json &generation_cfg, json &nntr_cfg);
  ~Lfm2VlVisionEncoder() override = default;

  void initialize() override;
  void load_weight(const std::string &base_path) override;

  /**
   * @brief Encode an in-memory FP32 image into LM-space embeddings.
   * @param image  multimodal_pointer{ float* pixels, byte_count }. Pixels are
   *               [3 * IMAGE_SIZE * IMAGE_SIZE] FP32 (CHW), preprocessed.
   * @return multimodal_pointer{ malloc'd float* embeds, n_img_tokens *
   *         out_features * sizeof(float) }. Caller takes ownership (free()).
   */
  multimodal_pointer run_image(const WSTR prompt, multimodal_pointer image,
                               int image_height, int image_width,
                               bool do_sample, const WSTR system_prompt,
                               const WSTR tail_prompt,
                               bool log_output) override;

private:
  json cfg_, generation_cfg_, nntr_cfg_;
  unsigned int downsample_factor_{2};
  std::unique_ptr<Lfm2VlVisionTransformer> vit_;
  std::unique_ptr<Lfm2VlConnector> connector_;
  std::string cache_dir_{"/data/local/tmp"}; /**< temp file dir for ViT input */
};

} // namespace causallm

#endif // __LFM2_VL_VISION_ENCODER_H__
```

- [ ] **Step 2: cpp 작성 — 생성/초기화/가중치 로드는 오케스트레이터 패턴 차용**

`lfm2_vl_vision_encoder.cpp`. `splitConfig`/키 이름은 `lfm2_vl_model.cpp`의 생성자·`load_weight`·`run`에서 사용하는 것과 **동일하게** 맞춘다(`vision_config`, `image_size`, `patch_size`, `hidden_size`, `vision_model_file`, `connector_model_file`):

```cpp
// SPDX-License-Identifier: Apache-2.0
#include "lfm2_vl_vision_encoder.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

namespace causallm {

static json pick(const json &cfg, const char *key) {
  return cfg.contains(key) ? cfg.at(key) : json::object();
}

Lfm2VlVisionEncoder::Lfm2VlVisionEncoder(json &cfg, json &generation_cfg,
                                         json &nntr_cfg)
  : Transformer(cfg, generation_cfg, nntr_cfg, ModelType::EMBEDDING),
    cfg_(cfg), generation_cfg_(generation_cfg), nntr_cfg_(nntr_cfg) {
  downsample_factor_ = cfg.value("downsample_factor", 2u);
  json vision_cfg = pick(cfg, "vision_config");
  if (vision_cfg.empty())
    vision_cfg = cfg; // flat vision-only config
  vit_ = std::make_unique<Lfm2VlVisionTransformer>(vision_cfg, generation_cfg,
                                                   nntr_cfg);
  unsigned int vit_embed = vision_cfg.value("hidden_size", 768u);
  unsigned int in_features =
    vit_embed * downsample_factor_ * downsample_factor_;
  unsigned int hidden = cfg.value("projector_hidden_size", 2560u);
  unsigned int out_features = cfg.value("text_hidden_size", 1024u);
  connector_ = std::make_unique<Lfm2VlConnector>(in_features, hidden,
                                                 out_features);
  if (nntr_cfg.contains("cache_dir"))
    cache_dir_ = nntr_cfg["cache_dir"].get<std::string>();
}

void Lfm2VlVisionEncoder::initialize() {
  vit_->initialize();
  vit_->allocateAndBindVitKVCache();
}

void Lfm2VlVisionEncoder::load_weight(const std::string &base_path) {
  // ViT weights
  if (nntr_cfg_.contains("vision_model_file"))
    vit_->load_weight(base_path + "/" +
                      nntr_cfg_["vision_model_file"].get<std::string>());
  else
    vit_->load_weight(base_path);
  // Connector weights
  if (nntr_cfg_.contains("connector_model_file"))
    connector_->loadWeights(
      base_path + "/" + nntr_cfg_["connector_model_file"].get<std::string>());
  else
    throw std::runtime_error(
      "Lfm2VlVisionEncoder: connector_model_file missing in nntr_config");
}

multimodal_pointer Lfm2VlVisionEncoder::run_image(
  const WSTR /*prompt*/, multimodal_pointer image, int /*image_height*/,
  int /*image_width*/, bool /*do_sample*/, const WSTR /*system_prompt*/,
  const WSTR /*tail_prompt*/, bool log_output) {

  json vision_cfg = pick(cfg_, "vision_config");
  if (vision_cfg.empty())
    vision_cfg = cfg_;
  unsigned int img_size   = vision_cfg.value("image_size", 256u);
  unsigned int patch_size = vision_cfg.value("patch_size", 16u);
  unsigned int vit_embed  = vision_cfg.value("hidden_size", 768u);
  unsigned int ph = img_size / patch_size;
  unsigned int pw = img_size / patch_size;
  unsigned int n_patches = ph * pw;

  // 1) Write incoming FP32 pixels [3*img*img] to a temp file the ViT reads.
  const size_t n_pixels = static_cast<size_t>(3) * img_size * img_size;
  std::string tmp_path = cache_dir_ + "/lfm2vl_vit_input.bin";
  {
    std::ofstream ofs(tmp_path, std::ios::binary);
    if (!ofs)
      throw std::runtime_error("Lfm2VlVisionEncoder: cannot open temp " +
                               tmp_path);
    ofs.write(reinterpret_cast<const char *>(image.first),
              static_cast<std::streamsize>(n_pixels * sizeof(float)));
  }

  // 2) Run ViT; features land in getLastFeatures().
  vit_->run(tmp_path, false, "", "", log_output);
  const std::vector<float> &feats = vit_->getLastFeatures();
  if (feats.empty())
    throw std::runtime_error("Lfm2VlVisionEncoder: ViT produced no features");

  // 3) pixel-unshuffle + connector MLP -> [n_img_tokens * out_features] FP32.
  auto unshuffled =
    pixelUnshuffle(feats, n_patches, vit_embed, ph, pw, downsample_factor_);
  unsigned int n_img_tokens = connector_->outTokens(n_patches);
  std::vector<float> embeds = connector_->forward(unshuffled, n_img_tokens);

  // 4) Hand back a malloc'd buffer (composer/api owns + frees it).
  const size_t out_bytes = embeds.size() * sizeof(float);
  void *out = std::malloc(out_bytes);
  if (!out)
    throw std::runtime_error("Lfm2VlVisionEncoder: OOM for embeds");
  std::memcpy(out, embeds.data(), out_bytes);
  if (log_output)
    std::cout << "[Lfm2VlVisionEncoder] img_tokens=" << n_img_tokens
              << " out_features=" << connector_->outFeatures() << "\n";
  return multimodal_pointer{out, out_bytes};
}

} // namespace causallm
```

- [ ] **Step 3: 빌드 시스템에 새 .cpp 등록**

`nntrainer/Applications/CausalLM/models/lfm2/lfm2-vl/meson.build`(없으면 상위 `models/meson.build`에서 lfm2-vl 소스 목록)을 열어 기존 `lfm2_vl_connector.cpp`/`lfm2_vl_model.cpp` 항목 옆에 `lfm2_vl_vision_encoder.cpp`를 추가. Android.mk 빌드도 쓰면 동일 글롭에 포함되는지 확인(메모리 `gauss-pluggable-bringup`의 generic `*/*.cpp` 글롭이면 자동 포함).

- [ ] **Step 4: 빌드 확인**

Run: 호스트 빌드(메모리 환경) → Expected: `lfm2_vl_vision_encoder.cpp` 컴파일·링크 성공.

- [ ] **Step 5: Commit**

```bash
cd nntrainer && git add Applications/CausalLM/models/lfm2/lfm2-vl/lfm2_vl_vision_encoder.{h,cpp} Applications/CausalLM/models/lfm2/lfm2-vl/meson.build
git commit -m "feat(lfm2-vl): add Lfm2VlVisionEncoder (ViT+connector -> run_image)"
```

### Task 1.3: 헤드리스 단독 테스트용 Factory 등록 (main.cpp)

**Files:**
- Modify: `nntrainer/Applications/CausalLM/main.cpp`

- [ ] **Step 1: include + 등록 추가**

`main.cpp` 상단 include에 `#include "models/lfm2/lfm2-vl/lfm2_vl_vision_encoder.h"` 추가. 기존 `Lfm2ForCausalLM` registerModel(295행 부근) 바로 아래에 추가:

```cpp
  causallm::Factory::Instance().registerModel(
    "Lfm2VlVisionEncoder", [](json cfg, json generation_cfg, json nntr_cfg) {
      return std::make_unique<causallm::Lfm2VlVisionEncoder>(
        cfg, generation_cfg, nntr_cfg);
    });
```

- [ ] **Step 2: 빌드 + 등록 확인**

Run: 호스트 빌드 후 `./quick_dot_ai_test` 인자 없이 실행 → Expected: `printRegistered`에 `Lfm2VlVisionEncoder` 포함.

- [ ] **Step 3: Commit**

```bash
cd nntrainer && git add Applications/CausalLM/main.cpp
git commit -m "feat(lfm2-vl): register Lfm2VlVisionEncoder in standalone factory"
```

---

## Milestone 2 — Quick.AI api: composer 개방 + 등록 + 마커 정합

### Task 2.1: api Factory에 LFM2 LM + vision encoder 등록

**Files:**
- Modify: `api/quick_dot_ai_api.cpp` (`register_models()`, ~257–317)

- [ ] **Step 1: include 추가**

파일 상단 모델 include 영역에:

```cpp
#include "lfm2_causallm.h"
#include "lfm2-vl/lfm2_vl_vision_encoder.h"
```

(경로는 api meson의 include_directories가 `models/lfm2`를 가리키는지 확인; 아니면 상대경로 `../nntrainer/Applications/CausalLM/models/lfm2/...`로 맞춘다.)

- [ ] **Step 2: `register_models()`의 `MultilingualTinyBert` 등록 뒤, `#ifdef ENABLE_QNN` 앞에 추가**

```cpp
    causallm::Factory::Instance().registerModel(
      "Lfm2ForCausalLM", [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Lfm2CausalLM>(cfg, generation_cfg,
                                                        nntr_cfg);
      });
    causallm::Factory::Instance().registerModel(
      "Lfm2VlVisionEncoder",
      [](json cfg, json generation_cfg, json nntr_cfg) {
        return std::make_unique<causallm::Lfm2VlVisionEncoder>(
          cfg, generation_cfg, nntr_cfg);
      });
```

- [ ] **Step 3: api 빌드 확인**

Run: `./build.sh --platform=android` (메모리 환경) → Expected: `libquick_dot_ai_api.so` 링크 성공(미정의 심볼 없음). 실패 시 api meson.build의 소스/링크에 lfm2 TU가 libcausallm로 들어오는지 확인.

- [ ] **Step 4: Commit**

```bash
git add api/quick_dot_ai_api.cpp
git commit -m "feat(api): register Lfm2ForCausalLM + Lfm2VlVisionEncoder in factory"
```

### Task 2.2: 공개 descriptor 2개 추가

**Files:**
- Modify: `api/model_descriptors_public.cpp`

- [ ] **Step 1: `kPublic[]` 배열의 `gemma4-cpu` 항목 뒤(QNN `#ifdef` 앞)에 추가**

```cpp
    {"lfm2-450m", "lfm2-vl", "LFM2-VL 450M (LM)", QDA_RUNTIME_NATIVE, B(0),
     QDA_CAP_STREAMING | QDA_CAP_MULTIMODAL,
     "LFM2-450M", /* device dir: /models/lfm2-450m */
     "Lfm2ForCausalLM"},
    {"siglip2-vl-encoder", "lfm2-vl", "SigLIP2 Vision Encoder",
     QDA_RUNTIME_NATIVE, B(0),
     QDA_CAP_VISION_ENCODER,
     "SIGLIP2-VL-ENCODER", /* device dir: /models/siglip2-vl-encoder */
     "Lfm2VlVisionEncoder"},
```

> `config_name`은 소문자화 시 디바이스 디렉터리명과 일치해야 한다(메모리: resolve_model_path가 lowercased config_name을 dir로 사용, quant suffix는 dead code). 따라서 디바이스 dir은 `lfm2-450m`, `siglip2-vl-encoder`.

- [ ] **Step 2: 빌드 + 카탈로그 확인**

Run: api 빌드 후 호스트/디바이스에서 `getModelCatalogJson()` 출력(또는 `api-app/test_api.cpp`로 카탈로그 덤프) → Expected: `lfm2-450m`(cap 0b0101=STREAMING|MULTIMODAL)과 `siglip2-vl-encoder`(cap 0b1000000=VISION_ENCODER) 등장.

- [ ] **Step 3: Commit**

```bash
git add api/model_descriptors_public.cpp
git commit -m "feat(api): add lfm2-450m + siglip2-vl-encoder public descriptors"
```

### Task 2.3: 멀티모달 경로 CPU 개방 + 이미지 마커 LFM2 정합 + 픽셀 레이아웃 일반화

**배경:** `execute_multimodal`/`run_vision_encoder`와 `runMultimodal*` 진입점이 `#ifdef ENABLE_QNN`로 가드돼 CPU 빌드에서 비활성. 또 마커가 `<|image|>` 하드코딩(LFM2는 `<image>`=396), `run_vision_encoder`가 `PATCH_SIZE=512`(512²) 하드코딩(LFM2는 256²).

**Files:**
- Modify: `api/quick_dot_ai_api.cpp`

- [ ] **Step 1: composer/헬퍼를 QNN 가드 밖으로 이동**

`#ifdef ENABLE_QNN`(2166) ... `#endif`(2289)로 감싼 `execute_multimodal`와 `run_vision_encoder`를 가드 **밖**으로 꺼낸다(가드 제거). 단 `run_vision_encoder` 내부에 QNN 전용 코드가 있으면 그 라인만 `#ifdef ENABLE_QNN`로 국소화. 또 `runMultimodalHandleStreaming`(2351), `runMultimodalHandleWithMessages`(2451) 등에서 `#ifdef ENABLE_QNN ... #else LOGE("built without ENABLE_QNN") #endif` 구조의 `#ifdef`/`#else`를 제거해 본문이 항상 컴파일되게 한다.

- [ ] **Step 2: 이미지 토큰 id 해석을 LFM2 호환으로**

`execute_multimodal`의 2188행을 교체:

```cpp
  // LFM2 uses "<image>" (id 396); gauss/vjepa use "<|image|>". Try both.
  int32_t image_token_id = tok->TokenToId("<|image|>");
  if (image_token_id < 0)
    image_token_id = tok->TokenToId("<image>");
```

- [ ] **Step 3: `run_vision_encoder` 픽셀 바이트 계산을 모델 주도로**

2280–2284행의 `PATCH_SIZE=512` 가정을 제거하고, 실제 이미지 해상도 기반으로:

```cpp
  // Pixel buffer is [3 * H * W] FP32 (CHW). The vision model interprets its
  // own expected resolution; we size the pointer by the given dimensions.
  const size_t pixel_bytes = static_cast<size_t>(3) *
                             static_cast<size_t>(originalHeight) *
                             static_cast<size_t>(originalWidth) * sizeof(float);
```

(기존 numPatches*3*512*512 라인 삭제. `numPatches` 인자는 LFM2 경로에서 미사용이나 시그니처는 유지.)

- [ ] **Step 4: CPU 빌드 확인**

Run: `./build.sh --platform=android` (ENABLE_QNN 없이도 빌드되는 native 변형 또는 QNN 빌드) → Expected: `runMultimodalHandleWithMessagesStreaming` 등이 LOGE 스텁이 아닌 실제 본문으로 컴파일.

- [ ] **Step 5: QNN 경로 무회귀 점검(스모크)**

기존 `vjepa-qnn`/gauss-vision 카탈로그·로드 경로가 컴파일·등록 그대로인지 확인(가드 제거가 QNN 전용 코드를 깨지 않았는지). 디바이스 QNN 빌드에서 V-JEPA 멀티모달 1회 로드까지.

- [ ] **Step 6: Commit**

```bash
git add api/quick_dot_ai_api.cpp
git commit -m "feat(api): enable multimodal composer on CPU + LFM2 image marker + model-driven pixel layout"
```

---

## Milestone 3 — 헤드리스 검증 (오라클 대조)

### Task 3.1: 디바이스 에셋 + nntr_config 준비

**Files:** (디바이스, 빌드 산출물 아님)

- [ ] **Step 1: LFM2-VL 가중치 변환 + 배치**

nntrainer `Applications/CausalLM/res/lfm2-vl/`의 변환 스크립트로 HF LFM2-VL-450M → nntrainer 바이너리 생성:
- `convert_vision_hf.py` → ViT 가중치, `convert_connector.py` → connector 가중치, `convert_lm.py` → LM 가중치, `convert_embedding.py` → standalone 임베딩 bin.

디바이스 배치:
```
/sdcard/Download/aistudio-mobile/models/siglip2-vl-encoder/  (ViT.bin, connector.bin, nntr_config.json)
/sdcard/Download/aistudio-mobile/models/lfm2-450m/           (lm.bin, tokenizer, embedding.bin, nntr_config.json)
```

- [ ] **Step 2: `nntr_config.json` 작성**

vision: `{"vision_model_file":"vit.bin","connector_model_file":"connector.bin","vision_config":{"image_size":256,"patch_size":16,"hidden_size":768},"projector_hidden_size":2560,"text_hidden_size":1024,"downsample_factor":2}`
LM: 기존 LFM2 LM 단독 config에 `embedding_bin_path`/`tokenizer_file`/`model_file_name` 포함(모놀리식 `res/lfm2-vl/README.md` 토큰 id·치수 참고: image=396,start=498,end=499, LM hidden=1024).

### Task 3.2: 모놀리식 오라클 출력 캡처

- [ ] **Step 1: 모놀리식 경로로 정답 생성**

전처리된 이미지 텐서(`naflex_preprocess.py`로 256² FP32 [3,256,256] 생성)와 프롬프트로 standalone 실행:

Run (디바이스 `/data/local/tmp/Quick.AI`): `./quick_dot_ai_test`에 `architecture=Lfm2VlForConditionalGeneration`, `image_tensor_path=<...>.bin`, `sample_input="What is in this image?"` 설정한 nntr_config로 실행.
Expected: 의미 있는 캡션/응답 + 생성 토큰열을 로그로 저장(오라클).

### Task 3.3: composer 페어 경로로 동일 입력 재현

- [ ] **Step 1: api 헤드리스로 vision+LLM 페어 로드 후 멀티모달 실행**

`api-app/test_api.cpp`(또는 동등 헤드리스 드라이버)에 케이스 추가: `loadMultimodalHandleByName("siglip2-vl-encoder", "lfm2-450m", ...)` → 전처리된 동일 이미지(256² FP32) + 프롬프트(`<image>` 포함)로 `runMultimodalHandleStreaming` 호출.

```cpp
// pseudo-driver (test_api.cpp): exact API per quick_dot_ai_api.h
CausalLmHandle h = nullptr;
loadMultimodalHandleByName(/*compute*/CPU, "siglip2-vl-encoder", "lfm2-450m",
                           model_base_path, native_lib_dir, &h);
runMultimodalHandleStreaming(h, "<image>What is in this image?",
                             pixels /*[3*256*256] FP32*/, /*numPatches*/256,
                             /*H*/256, /*W*/256, on_token, nullptr);
```

- [ ] **Step 2: 오라클 대조 (1차 게이트)**

Run: 위 드라이버 실행.
Expected: composer 경로 생성 토큰열이 Task 3.2 오라클과 **일치(또는 의미상 동일)**. 불일치 시 [[gemma4-qnn-garbage-debug]] 류 헤드리스 마커로 디버그:
- (a) `embeddingBytesPerToken`==4096, `lookupEmbedding(396)`이 nullptr 아님 확인.
- (b) `[MM] text=.. image=.. total=..` 로그에서 image 토큰 수 == connector `outTokens`(=256/4=64) 확인.
- (c) vision run_image 출력 임베딩의 스케일이 LM 임베딩과 동일 범위(FP32 identity)인지 — connector 출력 vs `lookupEmbedding`된 텍스트 임베딩의 norm 비교.
- (d) 마커 위치: `<image>`가 396으로 토큰화되어 splice 위치가 오케스트레이터(`<|image_start|><image><|image_end|>`)와 정합하는지 — 필요 시 프롬프트 템플릿/마커를 오케스트레이터와 동일하게.

- [ ] **Step 3: 검증 메모 커밋(있으면)**

```bash
git add api-app/test_api.cpp
git commit -m "test(api): headless LFM2-VL composer pair vs monolithic oracle"
```

---

## Milestone 4 — Android 앱: 믹스앤매치 picker + NaFlex 전처리

### Task 4.1: `loadMultimodalHandleByName` JNI 노출

**Files:**
- Modify: `Android/QuickDotAI/src/main/cpp/quickai_jni.cpp`
- Modify: `Android/QuickDotAI/src/main/java/com/example/quickdotai/NativeCausalLm.kt`

- [ ] **Step 1: JNI 함수 추가 (quickai_jni.cpp)**

기존 `loadModelHandleByName` JNI 구현 패턴을 따라 추가(핸들 long 반환):

```cpp
extern "C" JNIEXPORT jlong JNICALL
Java_com_example_quickdotai_NativeCausalLm_loadMultimodalHandleByNameNative(
    JNIEnv *env, jobject /*thiz*/, jint compute, jstring emb_id, jstring llm_id,
    jstring base_path, jstring native_lib_dir) {
  const char *c_emb = env->GetStringUTFChars(emb_id, nullptr);
  const char *c_llm = env->GetStringUTFChars(llm_id, nullptr);
  const char *c_base = env->GetStringUTFChars(base_path, nullptr);
  const char *c_nld =
      native_lib_dir ? env->GetStringUTFChars(native_lib_dir, nullptr) : nullptr;
  CausalLmHandle handle = nullptr;
  ErrorCode ec = loadMultimodalHandleByName(
      static_cast<BackendType>(compute), c_emb, c_llm, c_base, c_nld, &handle);
  env->ReleaseStringUTFChars(emb_id, c_emb);
  env->ReleaseStringUTFChars(llm_id, c_llm);
  env->ReleaseStringUTFChars(base_path, c_base);
  if (c_nld) env->ReleaseStringUTFChars(native_lib_dir, c_nld);
  if (ec != CAUSAL_LM_ERROR_NONE) {
    LOGE("loadMultimodalHandleByNameNative failed: %d", ec);
    return 0;
  }
  return reinterpret_cast<jlong>(handle);
}
```

> `loadMultimodalHandleByName`의 정확한 C 시그니처는 `api/quick_dot_ai_api.h:250` 확인 후 인자 순서를 맞춘다(compute/emb_id/llm_id/base/native_lib_dir/out_handle).

- [ ] **Step 2: Kotlin external fun + 래퍼 (NativeCausalLm.kt)**

기존 멀티모달 external fun 영역(272행 부근)에:

```kotlin
external fun loadMultimodalHandleByNameNative(
    compute: Int, embId: String, llmId: String,
    basePath: String, nativeLibDir: String?
): Long
```

그리고 공개 래퍼(기존 loadModelHandleByName 래퍼 옆):

```kotlin
fun loadMultimodalHandleByName(
    compute: BackendType, embId: String, llmId: String,
    basePath: String, nativeLibDir: String? = null
): Long = loadMultimodalHandleByNameNative(
    compute.ordinal, embId, llmId, basePath, nativeLibDir)
```

- [ ] **Step 3: 빌드 확인**

Run: `./Android/gradlew :QuickDotAI:assembleDebug` → Expected: BUILD SUCCESSFUL, JNI 심볼 링크.

- [ ] **Step 4: Commit**

```bash
git add Android/QuickDotAI/src/main/cpp/quickai_jni.cpp Android/QuickDotAI/src/main/java/com/example/quickdotai/NativeCausalLm.kt
git commit -m "feat(android): expose loadMultimodalHandleByName via JNI"
```

### Task 4.2: ModelCatalog 헬퍼 (vision/LLM 분류)

**Files:**
- Modify: `Android/QuickDotAI/src/main/java/com/example/quickdotai/ModelCatalog.kt`

- [ ] **Step 1: 헬퍼 추가**

`ModelCatalog` object에:

```kotlin
/** Vision-encoder producers selectable as the multimodal "eye". */
fun visionEncoders(): List<ModelDescriptor> =
    all().filter { Capability.VISION_ENCODER in it.capabilities }

/** LLMs that can consume image embeddings (MULTIMODAL-capable). */
fun pairableLlms(): List<ModelDescriptor> =
    all().filter { Capability.MULTIMODAL in it.capabilities }
```

> `Capability` enum에 `VISION_ENCODER`가 있는지 확인. 없으면 `ModelCatalog.kt`의 capability 비트 파서(104–111행)에 `0b1000000 -> VISION_ENCODER` 추가 + enum 보강(api `QDA_CAP_VISION_ENCODER = 1u<<6`와 정합).

- [ ] **Step 2: 빌드 + 단위 확인**

Run: `./Android/gradlew :QuickDotAI:compileDebugKotlin` → Expected: 성공. (가능하면 `visionEncoders()`가 `siglip2-vl-encoder`, `pairableLlms()`가 `lfm2-450m` 포함하는지 카탈로그 덤프 로그로 확인.)

- [ ] **Step 3: Commit**

```bash
git add Android/QuickDotAI/src/main/java/com/example/quickdotai/ModelCatalog.kt
git commit -m "feat(android): catalog helpers for vision encoders + pairable LLMs"
```

### Task 4.3: SigLIP-NaFlex 전처리(MVP 고정 256²)

**Files:**
- Create: `Android/QuickDotAI/src/main/java/com/example/quickdotai/SigLipNaFlexImageProcessor.kt`

- [ ] **Step 1: 프로세서 작성 (256² square, mean/std 0.5, CHW)**

`LlavaNextImageProcessor`의 `ModelInput`/정규화 패턴을 따르되 256²·patch16:

```kotlin
package com.example.quickdotai

import android.graphics.Bitmap
import android.graphics.Color

/**
 * SigLIP2 (LFM2-VL) preprocessing — MVP: fixed 256x256 square resize, no
 * NaFlex dynamic resolution. Output: FP32 CHW [3*256*256], normalized to
 * (x/255 - 0.5)/0.5.
 */
class SigLipNaFlexImageProcessor {
    companion object {
        const val IMAGE_SIZE = 256
        const val PATCH_SIZE = 16
        private const val MEAN = 0.5f
        private const val STD = 0.5f
    }

    /** Returns (pixelValues CHW FP32, numPatches=(IMAGE_SIZE/PATCH_SIZE)^2). */
    fun preprocess(src: Bitmap): NativeCausalLm.MultimodalInput {
        val resized = Bitmap.createScaledBitmap(src, IMAGE_SIZE, IMAGE_SIZE, true)
        val n = IMAGE_SIZE * IMAGE_SIZE
        val out = FloatArray(3 * n)
        val px = IntArray(n)
        resized.getPixels(px, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)
        for (i in 0 until n) {
            val p = px[i]
            out[i]         = ((Color.red(p)   / 255f) - MEAN) / STD // R plane
            out[n + i]     = ((Color.green(p) / 255f) - MEAN) / STD // G plane
            out[2 * n + i] = ((Color.blue(p)  / 255f) - MEAN) / STD // B plane
        }
        val patches = (IMAGE_SIZE / PATCH_SIZE) * (IMAGE_SIZE / PATCH_SIZE)
        return NativeCausalLm.MultimodalInput(
            pixelValues = out, numPatches = patches,
            originalHeight = IMAGE_SIZE, originalWidth = IMAGE_SIZE)
    }
}
```

> `MultimodalInput` 필드명/패키지는 `NativeCausalLm.kt:117` 확인 후 정합. `numPatches`는 헤드리스에서 검증한 vision encoder 기대값(=256, ViT 입력 패치 수)과 동일 의미인지 확인 — composer는 numPatches를 LFM2 경로에서 미사용하므로 값은 진단용.

- [ ] **Step 2: 빌드 확인**

Run: `./Android/gradlew :QuickDotAI:compileDebugKotlin` → Expected: 성공.

- [ ] **Step 3: Commit**

```bash
git add Android/QuickDotAI/src/main/java/com/example/quickdotai/SigLipNaFlexImageProcessor.kt
git commit -m "feat(android): SigLIP NaFlex image processor (MVP fixed 256)"
```

### Task 4.4: 믹스앤매치 picker UI + 페어 로드 + 전처리 분기

**Files:**
- Modify: `Android/SampleTestAPP/.../MainActivity.kt`

- [ ] **Step 1: vision/LLM 드롭다운 2개 추가**

OpenAI 탭(멀티모달 경로)에 기존 FAMILY 드롭다운 패턴(`dropdownField`)을 재사용해 두 개 추가:
- "VISION ENCODER" = `ModelCatalog.visionEncoders().map { it.id }`
- "LLM" = `ModelCatalog.pairableLlms().map { it.id }`
선택값을 상태로 보관(`var selectedVisionId by remember`, `var selectedLlmId by remember`).

- [ ] **Step 2: 페어 로드 버튼 동작**

"Load multimodal pair" 액션에서:

```kotlin
val handle = NativeCausalLm.loadMultimodalHandleByName(
    compute = selectedBackend, embId = selectedVisionId,
    llmId = selectedLlmId, basePath = modelBasePath,
    nativeLibDir = applicationInfo.nativeLibraryDir)
if (handle == 0L) { /* show error banner */ } else { currentHandle = handle }
```

- [ ] **Step 3: 전처리 분기 (모델별)**

이미지 픽 후 프로세서 선택:

```kotlin
val mmInput = if (selectedVisionId == "siglip2-vl-encoder")
    SigLipNaFlexImageProcessor().preprocess(bitmap)
else
    LlavaNextImageProcessor(...).process(bitmap)  // 기존 경로 유지
```

그리고 `runMultimodalHandleWithMessagesStreaming(handle, messages, ..., mmInput.pixelValues, mmInput.numPatches, mmInput.originalHeight, mmInput.originalWidth, listener)` 호출(프롬프트/메시지에 `<image>` 포함).

- [ ] **Step 4: 빌드 확인**

Run: `./Android/gradlew :SampleTestAPP:compileDebugKotlin` → Expected: BUILD SUCCESSFUL.

- [ ] **Step 5: Commit**

```bash
git add Android/SampleTestAPP
git commit -m "feat(app): vision+LLM mix-and-match picker + LFM2-VL preprocessing path"
```

---

## Milestone 5 — APK on-device 검증

### Task 5.1: APK 빌드·설치 + 페어 로드·추론

**절차:** 메모리 `quickai-android-build-env`(빌드/설치) + `gauss-pluggable-bringup`(APK verify: 탭별 로더, 케이스마다 force-stop+relaunch).

- [ ] **Step 1: 빌드·설치**

Run: `./build.sh --platform=android && ./apk_install_android.sh` (또는 메모리 기준 명령).
Expected: 설치 성공, 16KB 정렬 경고는 무해(OK).

- [ ] **Step 2: 디바이스 에셋 확인**

`/sdcard/Download/aistudio-mobile/models/siglip2-vl-encoder/`, `/lfm2-450m/`에 가중치+config 존재(Task 3.1과 동일).

- [ ] **Step 3: 믹스앤매치 로드 + 이미지 추론**

OpenAI 탭 → VISION="siglip2-vl-encoder", LLM="lfm2-450m" 선택 → "Load multimodal pair" → 이미지 픽 → 메시지(`<image>` 포함) → Run(streaming).
Expected: logcat `QuickAI`에 `SINGLE/MULTI-MODEL SUCCESS` + `[MM] text=.. image=64 total=..`, 콘솔에 이미지를 실제 인지한 일관 응답. gauss-vision "이미지 안 보임" 회귀 없는지 확인.

- [ ] **Step 4: 텍스트-only LFM2 LM 단독 점검(회귀)**

LLM만 `lfm2-450m`로 일반 텍스트 로드/생성도 정상인지(임베딩 가상 추가가 텍스트 경로 무영향) 확인.

- [ ] **Step 5: 결과를 메모리에 기록**

검증 매트릭스(디바이스/케이스/결과)를 메모리 `gauss-pluggable-bringup` 또는 신규 `lfm2-vl-bringup` 메모리에 추가.

---

## Self-Review (spec 대조)

- **분해(spec 4.1)** → Task 1.1/1.2/1.3 ✅ (vision encoder 래퍼 + LM base 가상 + Factory).
- **composer CPU 개방(spec 4.2)** → Task 2.3 Step 1/4 ✅.
- **이미지 마커 정합 396(spec 4.2)** → Task 2.3 Step 2 ✅.
- **splice 토큰 수 = outTokens(spec 4.2)** → 기존 composer가 `image_embeds.size()/bpt`로 자동 계산(Task 3.3 Step 2(b)에서 64 검증) ✅.
- **공개 descriptor 2개(spec 4.2)** → Task 2.2 ✅.
- **JNI loadMultimodalHandleByName(spec 4.3)** → Task 4.1 ✅.
- **믹스앤매치 UI(spec 4.3)** → Task 4.2 + 4.4 ✅.
- **NaFlex 전처리 MVP 256²(spec 4.3)** → Task 4.3 ✅.
- **헤드리스→APK 단계 검증(spec 6)** → Milestone 3 + 5 ✅.
- **리스크: 임베딩 공간 정합(spec 7)** → Task 3.3 Step 2(a)(c) 게이트 ✅.
- **리스크: 마커 의미론(spec 7)** → Task 3.3 Step 2(d) 오라클 대조 ✅.
- **리스크: QNN 무회귀(spec 7)** → Task 2.3 Step 5 ✅.

**미반영(의도적, spec 8 후속):** full NaFlex 동적 해상도, 다중 이미지, NPU 양자화, ViT in-memory 입력 경로(MVP는 temp 파일), `ModelDescriptor` image_token 스키마 정식화.

**알려진 불확정(구현 중 첫 빌드에서 확정):** (1) `DIM` 등 base hidden-size 멤버 정확명 — Task 1.1에서 concrete `lookupEmbedding` 본문으로 확인. (2) lfm2 헤더 include 경로(api meson include_directories) — Task 2.1 Step 1. (3) `MultimodalInput`/`Capability.VISION_ENCODER` 존재 — Task 4.2/4.3. (4) `loadMultimodalHandleByName` C 시그니처 인자 순서 — Task 4.1.
