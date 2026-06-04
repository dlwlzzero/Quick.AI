# Model Family 선택을 드롭다운으로 변경 — 설계 문서

- **날짜**: 2026-06-02
- **대상 앱**: Quick.AI Android (`SampleTestAPP`)
- **상태**: 승인됨, 구현 계획 대기

## 1. 목적

현재 모델 **FAMILY** 선택은 가로 스크롤 칩 행(`chipRow()`)으로 구현되어 있다. FAMILY 항목 수가 늘어나면서 칩 행이 길어지고 어떤 값이 선택됐는지 한눈에 파악하기 어렵다. FAMILY 선택을 Material 스타일 드롭다운 필드로 바꿔 선택값 가독성과 화면 공간 효율을 개선한다.

## 2. 범위

### 포함
- **FAMILY 선택만** 드롭다운으로 변경.
- Run/OpenAI 탭과 Chat 탭 **두 곳 모두** 적용.

### 제외 (변경하지 않음)
- RUNTIME / BACKEND / QUANTIZATION 선택 — 기존 `chipRow()` 칩 그대로 유지.
- FAMILY 변경 시 RUNTIME/BACKEND를 재계산하는 cascading 로직 — 기존 람다 그대로 재사용.
- `ModelCatalog.kt` — 옵션 소스(`ModelCatalog.families()`)는 변경 없이 그대로 사용.
- 자동화 UI 테스트 도입 — 프로젝트에 UI 테스트 인프라가 없으므로 범위 밖.

## 3. 아키텍처 / 접근

`MainActivity.kt`에 신규 헬퍼 함수 `dropdownField()`를 추가한다. 시그니처는 기존 `chipRow()`와 **동일**하게 맞춘다:

```kotlin
private fun dropdownField(
    t: M3Tokens,
    options: List<String>,
    selected: String,
    onPick: (String) -> Unit
): View
```

시그니처를 동일하게 맞추면 FAMILY 호출부 2곳에서 함수 이름만 `chipRow` → `dropdownField`로 교체하면 되고, cascading 동작을 담은 기존 람다는 그대로 전달된다.

```
chipRow()        ← RUNTIME / BACKEND / QUANTIZATION 계속 사용 (변경 없음)
dropdownField()  ← FAMILY 전용 (신규)
```

이 방식으로 변경 범위를 최소화하고 기존 칩 로직과의 결합을 끊는다.

## 4. `dropdownField()` 컴포넌트 설계

### 외형
- M3 토큰을 사용하는 outlined 필드.
- 좌측에 현재 선택값 텍스트(`selected`), 우측에 `▾` 아이콘.
- 테두리/배경은 기존 헬퍼(`strokedSolid`, `solid`, `dp`)와 색상 토큰(`onSurface`, `onSurfaceVar`, `outline` 등)을 재사용해 칩과 시각적 일관성 유지.

### 상호작용
- 필드를 탭하면 `android.widget.PopupMenu`를 필드 View에 앵커시켜 `options` 목록을 표시.
- 메뉴 항목 선택 시 `onPick(opt)` 호출 → 기존 람다가 상태를 갱신하고 `rebuildUi()`를 호출 → 필드 라벨이 새 선택값으로 다시 그려진다.

### 선택 표시
- 팝업 메뉴에서 현재 선택된 항목에 체크(✓) 표시(`MenuItem.setChecked`).

### 방어 처리
- `options`가 비어 있으면 필드를 비활성(회색) 상태로 표시하고 탭 동작을 막는다.

### PopupMenu 선택 이유
별도 레이아웃/어댑터 없이 View에 앵커되는 네이티브 팝업이라 코드량이 적다. Spinner와 달리 M3 색상 토큰과 충돌하는 OS 기본 스타일 박스가 없어 테마 일관성을 유지하기 쉽다.

## 5. 변경 지점

| 파일 / 위치 | 변경 내용 |
|-------------|-----------|
| `Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt` (`chipRow()` 정의부 ~1588 근처) | `dropdownField()` 헬퍼 신규 추가 |
| `MainActivity.kt:565` (Run/OpenAI 탭 FAMILY) | `chipRow(...)` → `dropdownField(...)` — 전달 람다 동일 |
| `MainActivity.kt:796` (Chat 탭 FAMILY) | `chipRow(...)` → `dropdownField(...)` — 전달 람다 동일 |

`ModelCatalog.kt`는 변경 없음.

### 참고: 현재 호출부 (변경 후 함수명만 교체)

Run/OpenAI 탭 (565–571):
```kotlin
body.addView(chipRow(t, ModelCatalog.families(), selFamily) { picked ->
    selFamily = picked
    selRuntime = ModelCatalog.runtimesFor(selFamily).firstOrNull() ?: selRuntime
    selBackend = ModelCatalog.backendsFor(selFamily, selRuntime).firstOrNull() ?: selBackend
    modelPathText = defaultModelPathFor(selDescriptor, selectedQuant) ?: ""
    rebuildUi(resetModelPath = true)
})
```

Chat 탭 (796–802):
```kotlin
modelCard.addView(chipRow(t, ModelCatalog.families(), chatSelFamily) { picked ->
    chatSelFamily = picked
    chatSelRuntime = ModelCatalog.runtimesFor(chatSelFamily).firstOrNull() ?: chatSelRuntime
    chatSelBackend = ModelCatalog.backendsFor(chatSelFamily, chatSelRuntime).firstOrNull() ?: chatSelBackend
    clearChatSessionState()
    rebuildUi()
})
```

각 호출의 `chipRow` → `dropdownField` 한 단어만 바뀐다.

## 6. 데이터 흐름

```
사용자 탭 → PopupMenu 표시(options = ModelCatalog.families())
        → 항목 선택 → onPick(picked)
        → (기존 람다) selFamily/chatSelFamily 갱신 + runtime/backend 재계산
        → rebuildUi() → dropdownField가 새 selected 값으로 다시 렌더
```

상태 변수(`selFamily`, `chatSelFamily`)와 cascading 재계산은 전혀 바뀌지 않는다. 드롭다운은 칩과 동일한 입력/출력 계약을 따르는 표현(presentation) 레이어 교체일 뿐이다.

## 7. 오류 처리 / 엣지 케이스

- **빈 옵션 목록**: 비활성 필드로 표시, 탭 무동작.
- **선택값이 옵션에 없음**: `selected` 텍스트를 그대로 표시(기존 칩과 동일하게 강제 변경하지 않음). cascading 기본값 로직이 이미 유효 값을 보장.
- **테마**: 라이트/다크 양쪽에서 토큰 기반 색상 사용으로 자동 대응.

## 8. 테스트 / 검증

프로젝트는 UI를 코드로 직접 생성하며 자동화 UI 테스트가 없다. 검증은 **빌드 + 디바이스 설치 후 수동 확인**으로 진행한다 (빌드 환경은 메모리 `quickai-android-build-env` 참고).

검증 항목:
1. Run/OpenAI 탭에 FAMILY 드롭다운이 표시되고 현재 선택값이 보인다.
2. Chat 탭에 FAMILY 드롭다운이 표시되고 현재 선택값이 보인다.
3. 드롭다운에서 다른 family 선택 시 RUNTIME/BACKEND 칩이 cascading으로 재계산된다 (회귀 없음).
4. RUNTIME/BACKEND/QUANTIZATION은 여전히 칩으로 표시된다.
5. 라이트/다크 테마에서 외형이 깨지지 않는다.

## 9. 미해결 / 향후 과제

- 향후 RUNTIME/BACKEND/QUANTIZATION도 드롭다운 통일을 원하면 동일 `dropdownField()` 재사용으로 확장 가능 (이번 범위 밖).
