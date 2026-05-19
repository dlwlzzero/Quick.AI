# Performance Metrics Calculation Fix Plan

## 1. Executive Summary

Quick.AI 프로젝트의 inference 성능 메트릭(속도, 지연 시간, 메모리 사용량) 계산 로직에 대한 분석 결과, **QNN(Qualcomm QNN) 기반 모델 전부에서 `performance_metrics` 구조체가 전혀 채워지지 않고 있거나, 완전히 누락되어 있습니다.** 기본 CausalLM(`causal_lm.cpp`)은 정상적으로 계산하고 있으나, QNN 모델들은 `Transformer::performance_metrics` 멤버에 값을 기록하지 않으며, 일부는 전역 변수를 사용하여 동시 실행 시 데이터 경쟁(race condition)이 발생할 수 있습니다.

또한 API 레이어(`quick_dot_ai_api.cpp`)의 `metrics_on_handle()` 함수가 멀티모달 모델(비전 인코더 + LLM) 구조를 처리하지 못해, 항상 `models[0]`(비전 인코더)의 빈 메트릭을 반환하는 버그가 있습니다.

---

## 2. Current State Analysis

### 2.1 Metric Definition

메트릭 정의는 `nntrainer/Applications/CausalLM/models/performance_metrics.h`에 위치합니다:

```cpp
typedef struct {
  unsigned int prefill_tokens;
  double prefill_duration_ms;
  unsigned int generation_tokens;
  double generation_duration_ms;
  double total_duration_ms;
  double initialization_duration_ms;
  size_t peak_memory_kb;
} TransformerPerformanceMetrics;
```

### 2.2 Metric Collection Point

메트릭은 각 모델의 `run()` 메서드 내부에서 수집되며, `Transformer` 베이스 클래스의 `performance_metrics` 멤버에 기록됩니다. API 레이어는 `getPerformanceMetrics()`를 통해 이 값을 읽어 C API로 내보냅니다.

### 2.3 Current Status by Model

| Model | File | `run()` Override | Sets `performance_metrics` | Status |
|-------|------|------------------|----------------------------|--------|
| **Base CausalLM** | `causal_lm.cpp` | Yes (base impl) | Yes | **OK** |
| **Gauss3_8_VIT_QNN** | `gauss3_8_vit_qnn.cpp` | Yes | Partial (hardcoded `_len=1`, `peak_memory_kb=0`) | **NEEDS FIX** |
| **Gauss3_6_QNN** | `gauss3_6_qnn.cpp` | Yes | **No** | **BROKEN** |
| **Gauss3_8_QNN** | `gauss3_8_qnn.cpp` | Yes | **No** | **BROKEN** |
| **Gemma4_E2B_QNN** | `gemma4_e2b_qnn.cpp` | Yes | **No** | **BROKEN** |
| **Gauss3_8_VEncoder_QNN** | `gauss3_8_vision_encoder_qnn.cpp` | Yes (empty) | **No** | **BROKEN** |

---

## 3. Detailed Issues

### 3.1 C API Layer: Multimodal Metrics Bug

**File:** `api/quick_dot_ai_api.cpp:1414`

**Problem:**
`metrics_on_handle()` always reads metrics from `h.models[0]`. For multimodal models like `Gauss3_8_VIT_QNN`, the model vector is structured as:
- `models[0]`: Vision Encoder (`Gauss_3_8_VEncoder_QNN`)
- `models[1]`: Text LLM (`Gauss_3_8_QNN`)

The API has a helper `text_generation_model_index()` that correctly identifies index 1 for text generation metrics, but `metrics_on_handle()` does not use it. As a result, calling `getPerformanceMetricsHandle()` on a VIT model returns zeros because `models[0]` never runs a text generation loop.

**Impact:** Android Metrics tab shows all zeros for any multimodal inference.

**Fix:**
Change `metrics_on_handle()` to use `text_generation_model_index(h)` instead of hardcoded `0` when fetching the model for metrics.

---

### 3.2 Base CausalLM: Working Reference

**File:** `nntrainer/Applications/CausalLM/models/causal_lm.cpp:605-632`

The base CausalLM correctly:
1. Measures prefill start/end with `std::chrono::high_resolution_clock`
2. Measures generation start/end separately
3. Measures total duration from `run()` entry to exit
4. Calls `getPeakMemoryKb()` for peak memory
5. Populates all fields of `performance_metrics`
6. Sets `has_run_ = true`

This serves as the **canonical implementation** that QNN models must replicate.

---

### 3.3 Gauss3_8_VIT_QNN: Partially Working

**File:** `src/models/qnn/gauss-3.8-vit-qnn/gauss3_8_vit_qnn.cpp:408-416`

**Problems:**
1. `prefill_tokens = _len` where `_len` is hardcoded to `1` (TODO comment indicates this is a placeholder)
2. `peak_memory_kb = 0` with a TODO comment
3. No `initialization_duration_ms` is recorded (this is set at the API layer, so the model itself doesn't need to)

**Fix:**
- Replace hardcoded `_len` with actual token count from input
- Add `getPeakMemoryKb()` call

---

### 3.4 Gauss3_6_QNN: Completely Missing Metrics

**File:** `src/models/qnn/gauss-3.6-qnn/gauss3_6_qnn.cpp`

**Problems:**
1. `performance_metrics` struct is **never populated**
2. Uses global variable `raw_exec_seconds` (defined in `generate_qnn_utils.h`) for timing
3. Timer starts **after** prefill, so prefill duration is lost
4. Token count calculation uses `idx - input_len`, which is incorrect when continuing a conversation (KV cache warm). Should be `idx - prefill_len`
5. Division-by-zero risk in console output: `raw_exec_seconds.count() / (idx - input_len)` with no guard
6. `has_run_` is set to `true` but metrics remain zero

**Fix:**
- Add `performance_metrics` population at end of `run()`
- Move `raw_exec_seconds` from global to instance member variable
- Start prefill timer before prefill inference call
- Start generation timer after prefill ends
- Fix token count formula
- Add `std::max(1, ...)` guard for division

---

### 3.5 Gauss3_8_QNN: Completely Missing Metrics

**File:** `src/models/qnn/gauss-3.8-qnn/gauss3_8_qnn.cpp`

**Problems:**
1. `performance_metrics` struct is **never populated** in either `run()` or `run_with_embeddings()`
2. Uses global `raw_exec_seconds`
3. Timer starts after prefill (prefill time lost)
4. `stop_requested_` handling exits loop without recording partial metrics

**Fix:**
- Populate `performance_metrics` in both `run()` and `run_with_embeddings()`
- Replace global timer with instance member
- Measure prefill separately
- Ensure partial metrics are recorded even on cancellation

---

### 3.6 Gemma4_E2B_QNN: Completely Missing Metrics

**File:** `src/models/qnn/gemma4-e2b-qnn/gemma4_e2b_qnn.cpp`

**Problems:**
1. `performance_metrics` struct is **never populated**
2. Uses global `raw_exec_seconds`
3. Timer starts after prefill (prefill time lost)
4. **`final_logit_softcapping` is forcibly set to 0.0f during initialization**, silently disabling a config feature (separate behavior bug)

**Fix:**
- Populate `performance_metrics`
- Replace global timer with instance member
- Measure prefill separately
- Fix `final_logit_softcapping` reset (if intentional, document)

---

### 3.7 Gauss3_8_Vision_Encoder_QNN: No Timing Logic

**File:** `src/models/qnn/gauss-3.8-vit-qnn/gauss3_8_vision_encoder_qnn.cpp`

**Problems:**
1. `run()` is **empty** (no-op)
2. `run_image()` is the actual inference path but has **no timing logic whatsoever**
3. No `performance_metrics` population

**Fix:**
- Add timing logic around `run_image()` inference call
- Populate `performance_metrics` (note: vision encoder "tokens" may need special handling — perhaps use number of image patches or feature tokens)
- Alternatively, if this model is always paired with a text model, ensure API layer reads from the text model's index

---

## 4. Implementation Plan

### Phase 1: API Layer Hotfix (Quick Win)

**Goal:** Fix multimodal metric bug in the C API.

**Tasks:**
1. [ ] Modify `metrics_on_handle()` in `api/quick_dot_ai_api.cpp`
   - Change: `auto *model = h.models[0].get();`
   - To: `auto *model = h.models[text_generation_model_index(h)].get();`

**Estimated LOE:** Very Low (~5 lines)

---

### Phase 2: QNN Model Metrics Implementation (Core Work)

**Goal:** Ensure all QNN models populate `performance_metrics` correctly.

#### 2-A: Common Infrastructure

**File:** `src/models/qnn/quick_dot_ai_qnn.h` or new utility

**Tasks:**
1. [ ] Add `raw_exec_seconds` as `std::chrono::duration<double>` member to `Quick_Dot_AI_QNN` base class (or per-model if they don't share a common base beyond Transformer)
2. [ ] Optionally create a helper macro or inline function `record_metrics(...)` to reduce duplication

#### 2-B: Gauss3_6_QNN

**File:** `src/models/qnn/gauss-3.6-qnn/gauss3_6_qnn.cpp`

**Tasks:**
1. [ ] Add `auto start_prefill = std::chrono::high_resolution_clock::now();` before prefill inference
2. [ ] Add `auto end_prefill = ...;` after prefill
3. [ ] Add `auto start_gen = ...;` before generation loop
4. [ ] Add `auto end_gen = ...;` after generation loop
5. [ ] Populate `performance_metrics`:
   ```cpp
   performance_metrics.prefill_tokens = prefill_len;
   performance_metrics.prefill_duration_ms = prefill_ms;
   performance_metrics.generation_tokens = (std::max)(0U, (unsigned int)(idx - prefill_len));
   performance_metrics.generation_duration_ms = gen_ms;
   performance_metrics.total_duration_ms = prefill_ms + gen_ms;
   performance_metrics.peak_memory_kb = getPeakMemoryKb();
   ```
6. [ ] Add division-by-zero guard in console output
7. [ ] Remove/reference global `raw_exec_seconds` (use instance member)

#### 2-C: Gauss3_8_QNN

**File:** `src/models/qnn/gauss-3.8-qnn/gauss3_8_qnn.cpp`

**Tasks:**
1. [ ] Same timing instrumentation as 2-B for `run()`
2. [ ] Same timing instrumentation for `run_with_embeddings()`
3. [ ] Populate `performance_metrics` in both paths
4. [ ] Ensure partial metrics recorded on `stop_requested_` cancellation

#### 2-D: Gemma4_E2B_QNN

**File:** `src/models/qnn/gemma4-e2b-qnn/gemma4_e2b_qnn.cpp`

**Tasks:**
1. [ ] Same timing instrumentation as 2-B
2. [ ] Populate `performance_metrics`
3. [ ] Investigate/document `final_logit_softcapping = 0.0f` reset

#### 2-E: Gauss3_8_VIT_QNN

**File:** `src/models/qnn/gauss-3.8-vit-qnn/gauss3_8_vit_qnn.cpp`

**Tasks:**
1. [ ] Replace hardcoded `_len` with actual encoded token length
2. [ ] Add `peak_memory_kb = getPeakMemoryKb()`
3. [ ] Verify prefill/generation token counts are correct

#### 2-F: Gauss3_8_Vision_Encoder_QNN

**File:** `src/models/qnn/gauss-3.8-vit-qnn/gauss3_8_vision_encoder_qnn.cpp`

**Tasks:**
1. [ ] Add timing to `run_image()`
2. [ ] Define what "prefill_tokens" means for a vision encoder (image patch count?)
3. [ ] Populate `performance_metrics`

**Estimated LOE:** Medium (per model modification)

---

### Phase 3: Cleanup and Hardening

**Goal:** Remove technical debt and add safety guards.

**Tasks:**
1. [ ] **Remove global `raw_exec_seconds`** from `generate_qnn_utils.h` if no longer used by any model
2. [ ] **Add `std::max(1, ...)` guards** in all QNN console output for TPS division
3. [ ] **Fix Gauss3_6 token count** formula from `idx - input_len` to `idx - prefill_len`
4. [ ] **Verify `has_run_ = true`** is set in all QNN model run paths
5. [ ] **Add unit test** or integration test that loads each QNN model type, runs inference, and asserts `performance_metrics.generation_tokens > 0`

**Estimated LOE:** Low

---

## 5. Testing Plan

### 5.1 Unit Tests (C++ API Test)

**File:** `api-app/test_api.cpp` (extend)

1. After inference, assert `metrics.prefill_tokens > 0` (or `>=` for edge cases)
2. After inference, assert `metrics.generation_tokens > 0`
3. Assert `metrics.generation_duration_ms > 0`
4. Assert `metrics.total_duration_ms >= metrics.prefill_duration_ms + metrics.generation_duration_ms`

### 5.2 Manual Verification

For each QNN model type (`gauss3.6-qnn`, `gauss3.8-qnn`, `gemma4-e2b-qnn`):
1. Run `test_api` binary
2. Verify console output shows non-zero tokens and TPS
3. Verify Android Metrics tab shows correct values

### 5.3 Regression Test

1. Ensure base CausalLM metrics remain unchanged
2. Ensure Android JNI layer still correctly maps metrics through `NativeCausalLm$MetricsResult`

---

## 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **Breaking existing Android UI** if metric struct layout changes | Low | High | Do not change struct layout; only populate existing fields |
| **QNN model timing overhead** from additional `chrono` calls | Very Low | Low | `chrono` calls are nanosecond-scale overhead |
| **Incorrect token count** for VIT/vision encoder | Medium | Medium | Define semantics clearly (image patches vs tokens) |
| **Concurrent run() calls** clobbering metrics | Medium | Medium | Cannot happen per-handle (mutex), but global `raw_exec_seconds` must be removed |
| **Memory tracking not working** on QNN backends | Medium | Medium | `getPeakMemoryKb()` uses `rusage` which works at process level; acceptable for now |

---

## 7. Success Criteria

- [ ] `getPerformanceMetricsHandle()` returns non-zero metrics for all model types
- [ ] Multimodal models (VIT) return metrics from the text generation sub-model
- [ ] No global variables used for timing in any model
- [ ] Prefill and generation phases are separately timed in all models
- [ ] No division-by-zero crashes in console output
- [ ] `test_api` passes with metric validation assertions

---

## 8. Appendices

### A. Reference Implementation (from `causal_lm.cpp`)

```cpp
auto start_prefill = std::chrono::high_resolution_clock::now();
// ... prefill inference ...
auto finish_prefill = std::chrono::high_resolution_clock::now();
auto prefill_duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish_prefill - start_prefill);

auto start_generation = std::chrono::high_resolution_clock::now();
// ... generation loop ...
auto finish_generation = std::chrono::high_resolution_clock::now();
auto generation_duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish_generation - start_generation);

auto finish_total = std::chrono::high_resolution_clock::now();
auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(finish_total - start_total);
size_t peak_memory = getPeakMemoryKb();

performance_metrics.prefill_tokens = init_len;
performance_metrics.prefill_duration_ms = prefill_duration.count();
performance_metrics.generation_tokens = generation_cnt;
performance_metrics.generation_duration_ms = generation_duration.count();
performance_metrics.total_duration_ms = total_duration.count();
performance_metrics.peak_memory_kb = peak_memory;
has_run_ = true;
```

### B. Files to Modify

| Priority | File | Phase |
|----------|------|-------|
| Critical | `api/quick_dot_ai_api.cpp` | 1 |
| High | `src/models/qnn/gauss-3.6-qnn/gauss3_6_qnn.cpp` | 2 |
| High | `src/models/qnn/gauss-3.8-qnn/gauss3_8_qnn.cpp` | 2 |
| High | `src/models/qnn/gemma4-e2b-qnn/gemma4_e2b_qnn.cpp` | 2 |
| Medium | `src/models/qnn/gauss-3.8-vit-qnn/gauss3_8_vit_qnn.cpp` | 2 |
| Medium | `src/models/qnn/gauss-3.8-vit-qnn/gauss3_8_vision_encoder_qnn.cpp` | 2 |
| Low | `src/models/qnn/generate_qnn_utils.h` | 3 |
| Low | `api-app/test_api.cpp` | 3 |

---

*Plan created: 2026-05-19*
*Analyst: opencode*
