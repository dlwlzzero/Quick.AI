# LFM2-VL(SigLIP + LFM2) — Quick.AI pluggable composer 통합 설계 문서

- **날짜**: 2026-06-05
- **브랜치**: `v0.4.0` (nntrainer 서브모듈은 LFM2-VL 커밋 `0b52d15` 체크아웃 상태)
- **대상**: nntrainer 서브모듈 · Quick.AI public API(`libquick_dot_ai_api.so`) · Android(`QuickDotAI` AAR + `SampleTestAPP`)
- **상태**: 승인됨, 구현 계획 대기
- **분류**: 내부 문서 (`docs/superpowers/`는 git-ignore)

## 1. 배경 / 문제

nntrainer 서브모듈에 **LFM2-VL-450M**(이미지+텍스트 → 텍스트) 멀티모달 모델이
추가되었다. 구조는 다음 3개 컴포넌트로 이루어진 **모놀리식 오케스트레이터**다:

- `Lfm2VlVisionTransformer` (SigLIP2 ViT) — `models/lfm2/lfm2-vl/vision/lfm2_vl_vision_transformer.{h,cpp}`, arch `"Lfm2VlVisionTransformer"`. 비인과/RoPE 없음, patch16, NaFlex 지원.
- `Lfm2VlConnector` (pixel-unshuffle ×2 + LayerNorm + FC→GELU→FC, 3072→2560→1024) — `models/lfm2/lfm2-vl/lfm2_vl_connector.{h,cpp}`. Transformer 서브클래스가 **아님**(독립 MLP 클래스).
- `Lfm2CausalLM` (hybrid conv/attn LM, hidden 1024) — `models/lfm2/lfm2_causallm.{h,cpp}`. `lookupEmbedding()`(FP32/Q4_0/Q6_K), `run_with_embeddings()` 이미 오버라이드.

이 셋을 묶는 `Lfm2VlForConditionalGeneration`(`models/lfm2/lfm2-vl/lfm2_vl_model.{h,cpp}`,
arch `"Lfm2VlForConditionalGeneration"`)이 `run()` 안에서 ViT 인코딩 →
pixelUnshuffle → connector → `image_token_id=396` 위치에 임베딩 splice →
`run_with_embeddings()`까지 모두 수행한다. **문제는 이 경로가 nntrainer
standalone `main.cpp`(391–409행)의 직접 분기에서만 동작**하며, Factory 등록도
없고 Quick.AI API/앱에서는 **전혀 사용할 수 없다**는 점이다. CPU/FP32 모델이다.

한편 Quick.AI public API에는 이미 **generic 멀티모달 composer**가 있다:
`execute_multimodal()`(`api/quick_dot_ai_api.cpp:2175`)가
`[vision producer = models[0], LLM consumer = models[1]]` 페어를 받아, 텍스트
토큰 스트림에서 이미지 마커를 찾아 vision 임베딩을 splice하고
`llm->run_with_embeddings()`로 생성한다. 그러나 이 composer와 모든
`runMultimodal*` 진입점이 **`#ifdef ENABLE_QNN`로 가드**되어 있어(V-JEPA·gauss-vision은
NPU) CPU 전용 LFM2-VL에는 경로가 열리지 않는다. 또한 composer가 찾는 마커는
하드코딩된 `<|image|>`(`:2188`)인데 LFM2는 `image_token_id=396`(+start 498,
end 499)을 쓴다.

앱(`SampleTestAPP`)은 카탈로그 기반으로 동작한다: `nativeQueryCatalog()` JSON →
`ModelCatalog.kt` → `QDA_CAP_MULTIMODAL` 비트가 있으면 이미지 피커 +
`runMultimodal*Streaming` JNI 경로 자동 활성. 단 전처리는 LLaVA-Next식
(`LlavaNextImageProcessor.kt`, 512² 크롭, mean/std 0.5)이라 SigLIP2-NaFlex
(256², patch16) 규격과 맞지 않는다. 또 두 모델을 페어로 로드하는
`loadMultimodalHandleByName(emb_id, llm_id)`는 **JNI에 노출되어 있지 않다**.

## 2. 목표 / 범위

### 포함
- 모놀리식 LFM2-VL을 **두 개의 독립 Factory 모델**로 분해:
  (A) `Lfm2VlVisionEncoder` = ViT + connector를 묶어 `run_image()` 출력이 **LM
  임베딩 공간(1024-dim)** 으로 나오는 vision 인코더, (B) `Lfm2CausalLM` 단독 LM.
- 기존 generic `execute_multimodal` composer가 이 페어를 구동하도록
  **CPU(native) 빌드에서 컴파일/동작**하게 가드 개방.
- 이미지 마커 정합: composer가 하드코딩 `<|image|>` 대신 **LLM 토크나이저/디스크립터에서
  image token id를 조회**(LFM2=396).
- 앱에 **vision encoder + LLM 믹스앤매치 picker** UI 추가, `loadMultimodalHandleByName`
  JNI 노출.
- **SigLIP2-NaFlex 전처리** Kotlin 프로세서 추가(MVP: 고정 256² square).
- 검증: **헤드리스 api 테스트 → APK on-device** 단계적.

### 제외 (이번 범위 밖)
- 모놀리식 `Lfm2VlForConditionalGeneration` 자체 삭제/리팩터 — **레퍼런스로
  보존**(헤드리스 정답 대조용).
- full NaFlex 동적 해상도 + 위치 임베딩 보간 — **후속 단계**(MVP는 고정 해상도).
- QNN/NPU 버전의 LFM2-VL — 현재 모델은 CPU/FP32. NPU 양자화는 별도 과제.
- 다중 이미지(multi-image) LFM2-VL — 단일 이미지 우선.

## 3. 아키텍처 / 접근

핵심: **"한 덩어리 LFM2-VL을 vision(눈+변환기) · LLM(두뇌) 두 조각으로 분해 →
기존 pluggable composer로 페어링 → 앱에서 골라 끼움"**. API는 gauss-agnostic을
유지하며, LFM2-VL은 **공개 nntrainer 코드**이므로 래퍼는 nntrainer 서브모듈에
두고 Quick.AI는 **공개 descriptor만** 추가한다.

```
[App] vision picker (SigLIP) ─┐
      LLM picker (LFM2)     ─┴─► loadMultimodalHandleByName(emb_id, llm_id)
                                   │
[API] composer (CPU 개방) ──────────┤  models[0]=Lfm2VlVisionEncoder
      execute_multimodal           │  models[1]=Lfm2CausalLM
        ├─ run_vision_encoder ──────┤   ↳ run_image → ViT→unshuffle→connector → 1024-dim emb
        ├─ 이미지 토큰 id 조회(396) ─┤   ↳ image marker 위치 splice
        └─ llm->run_with_embeddings ┘   ↳ LFM2 디코딩
```

## 4. 변경 지점

### 4.1 nntrainer — vision 인코더 래퍼 + LM Factory 등록

- **신규 `Lfm2VlVisionEncoder`** (`models/lfm2/lfm2-vl/`): 내부에
  `Lfm2VlVisionTransformer` + `Lfm2VlConnector`를 소유. `run_image(pixels, …)`
  오버라이드 → ViT.run() → getLastFeatures() → pixelUnshuffle() →
  connector.forward() → **1024-dim `multimodal_pointer` 반환**. `get_embedding_info()`로
  LFM2 LM의 임베딩 양자화 스케일/오프셋과 정합. 결정 사항: **기존
  `Lfm2VlVisionTransformer`/`Lfm2VlConnector`를 직접 수정하지 않고 새 래퍼에서
  조합**(기존 모놀리식 경로 무손상). Factory 등록 arch `"Lfm2VlVisionEncoder"`,
  capability `QDA_CAP_VISION_ENCODER`.
- **`Lfm2CausalLM` Factory 등록 추가**: arch `"Lfm2ForCausalLM"`(또는 config의
  실제 architectures 문자열). 단독 텍스트 LLM으로도 로드 가능. `lookupEmbedding`/
  `run_with_embeddings`는 구현 완료 → 추가 작업 없음.
- 오케스트레이터(`Lfm2VlForConditionalGeneration`) + `main.cpp` 분기는 무변경.

### 4.2 Quick.AI API — composer CPU 개방 + 마커 정합 + descriptor

| 위치 | 변경 |
|------|------|
| `api/quick_dot_ai_api.cpp` `execute_multimodal`/`run_vision_encoder`/`runMultimodal*` | `#ifdef ENABLE_QNN` 가드에서 분리 → native(CPU) 빌드에서도 활성 |
| `api/quick_dot_ai_api.cpp:2188` 이미지 마커 | 하드코딩 `<|image|>` → **토크나이저/디스크립터에서 image token id 조회**(LFM2=396); start/end(498/499) 마커는 프롬프트 템플릿에서 처리 |
| splice 토큰 개수 | `connector.outTokens()`(= n_patches / r²)만큼 슬롯, vision 출력 길이 그대로 사용 |
| `api/model_descriptors_public.cpp` | 공개 descriptor 2개 추가: `siglip2-vl-encoder`(VISION_ENCODER), `lfm2-450m`(LM, MULTIMODAL 짝). config_name = 디바이스 dir과 일치(소문자 규칙). |

`ModelDescriptor`에 `image_token`(문자열 또는 id) 필드를 추가할지, 토크나이저
special token에서 유도할지는 구현 계획에서 확정(토크나이저 유도가 descriptor
스키마 변경을 피해 우선).

### 4.3 Android AAR / 앱 — 믹스앤매치 picker + NaFlex 전처리

- **JNI 노출** (`Android/QuickDotAI/.../quickai_jni.cpp` + `NativeCausalLm.kt`):
  `loadMultimodalHandleByName(emb_id, llm_id)` external fun 추가(현재 미노출).
- **믹스앤매치 UI** (`SampleTestAPP MainActivity.kt`): 카탈로그에서 capability로
  필터 — VISION_ENCODER 목록을 vision 드롭다운, 생성 가능 LLM 목록을 LLM
  드롭다운으로 노출. 두 값 선택 시 `loadMultimodalHandleByName`로 페어 로드.
  `ModelCatalog.kt`에 `visionEncoders()` / `pairableLlms()` 헬퍼 추가.
- **신규 `SigLipNaFlexImageProcessor.kt`**: SigLIP2 규격(256×256, patch16,
  mean/std 0.5, CHW). MVP는 **고정 256² square 리사이즈**. 모델 id로
  `LlavaNextImageProcessor`와 분기. `MultimodalInput`(pixelValues/numPatches/
  dims)에 맞춰 출력.

## 5. 데이터 흐름

```
[디바이스] /sdcard/Download/aistudio-mobile/models/<siglip-dir>/  (ViT+connector 가중치)
                                                 /<lfm2-dir>/      (LFM2 LM 가중치)
[앱] 이미지 픽 → SigLipNaFlexImageProcessor → pixelValues(256²,CHW) + numPatches
        vision=siglip2-vl-encoder, llm=lfm2-450m 선택
          └─ loadMultimodalHandleByName(emb,llm)  → 한 handle (models[0]=ViT래퍼, models[1]=LM)
              └─ runMultimodalHandleWithMessagesStreaming(...)
                  └─ [API] run_vision_encoder → 1024-dim emb
                       → 토큰화 후 image token(396) 위치에 splice
                       → llm->run_with_embeddings → 스트리밍 토큰 콜백
```

## 6. 단계적 검증

1. **헤드리스 api 테스트** (`quick_dot_ai_test`, 빌드 환경 메모리
   `quickai-android-build-env` 참고): vision 인코더 + LFM2 LM 페어 로드,
   **전처리된 이미지 텐서 + 프롬프트** 입력 → 출력이 nntrainer 모놀리식
   `Lfm2VlForConditionalGeneration` 레퍼런스(동일 이미지/프롬프트)와 **토큰 단위
   일치** 확인. connector 출력의 임베딩 양자화 정합이 1차 게이트.
2. **APK on-device** (S26 Ultra 등, 절차 메모리 `gauss-pluggable-bringup`의 APK
   verify 참고): 믹스앤매치로 페어 로드 → Chat/OpenAI 탭에서 이미지+질문 →
   이미지를 실제로 인지한 **일관된 응답** 확인(gauss-vision의 "이미지 안 보임"
   회귀 여부 점검).

## 7. 주요 리스크

- **임베딩 공간 정합**: connector 출력이 LFM2 LM 임베딩의 양자화 스케일/오프셋과
  정확히 맞지 않으면 의미 손실(gauss-vision "이미지 안 보임"류). 헤드리스 1차
  게이트로 조기 차단.
- **이미지 마커 의미론**: 396 위치 splice만으로 부족하고 start/end(498/499) 또는
  특정 프롬프트 템플릿이 필요할 수 있음 — 모놀리식 `run()`의 splice 로직
  (lfm2_vl_model.cpp:238–254)을 정답으로 대조.
- **NaFlex 전처리 정밀도**: 고정 256² MVP는 비정사각 이미지에서 정확도 하락 가능 —
  후속 동적 해상도 단계로 분리.
- **CPU 멀티모달 가드 개방의 부작용**: 기존 QNN 경로(V-JEPA/gauss-vision)
  무회귀 확인 필요.

## 8. 미해결 / 향후 과제

- full NaFlex 동적 해상도 + 위치 임베딩 보간(nntrainer `naflex_preprocess.py`/
  `naflex_interp` 유닛테스트 참고).
- LFM2-VL 다중 이미지 지원.
- NPU/QNN 양자화 버전.
- `ModelDescriptor`에 image_token/페어 힌트 스키마를 정식 추가할지 여부.
