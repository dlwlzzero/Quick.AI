# Model FAMILY Dropdown Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 모델 FAMILY 선택을 가로 스크롤 칩(`chipRow`)에서 Material 스타일 드롭다운 필드(`dropdownField`)로 교체한다. Run/OpenAI 탭과 Chat 탭 두 곳에 적용한다.

**Architecture:** `chipRow()`와 동일한 시그니처를 가진 신규 헬퍼 `dropdownField()`를 추가한다. FAMILY 호출부 2곳에서 함수명만 `chipRow` → `dropdownField`로 교체하며, FAMILY 변경 시 RUNTIME/BACKEND를 재계산하는 cascading 람다는 그대로 재사용한다. RUNTIME/BACKEND/QUANTIZATION 및 `ModelCatalog.kt`는 변경하지 않는다.

**Tech Stack:** Kotlin, 클래식 Android View (코드로 직접 UI 생성), Material 3 토큰 시스템, `android.widget.PopupMenu`. 빌드: Gradle (AGP 8.9.1, JDK 17).

**참고 스펙:** `docs/superpowers/specs/2026-06-02-family-dropdown-design.md`

---

## 빌드/검증 환경 (메모리 `quickai-android-build-env` 요약)

자동화 UI 테스트가 없으므로 각 코드 태스크는 **Kotlin 컴파일 통과**로 검증하고, 마지막에 **디바이스 설치 후 수동 확인**한다.

태스크 실행 전 셸에 아래 환경 변수를 export 해야 한다 (없으면 빌드 실패):

```bash
export NDK_ROOT=/home/jiyoung/Android/Sdk/ndk/android-ndk-r26b
export JAVA_HOME=/home/jiyoung/jdks/jdk-17.0.19+10
export ANDROID_HOME=/home/jiyoung/Android/Sdk
export ANDROID_SERIAL=R3CX80H8Y0F   # SM-S936U (S25+). minSdk=33 충족 기기만.
```

빠른 컴파일 검증 명령 (네이티브 빌드 없이 Kotlin만):

```bash
./Android/gradlew :SampleTestAPP:compileDebugKotlin
```

---

## File Structure

- **Modify only:** `Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt`
  - import 블록: `android.widget.PopupMenu` 추가
  - `chipRow()` 정의부(~1588) 근처: `dropdownField()` 신규 추가
  - `:565` Run/OpenAI 탭 FAMILY 호출: `chipRow` → `dropdownField`
  - `:796` Chat 탭 FAMILY 호출: `chipRow` → `dropdownField`
- **변경 없음:** `Android/QuickDotAI/.../ModelCatalog.kt`, 기타 모든 파일

참고로 이 파일에는 이미 다음 헬퍼/임포트가 존재하여 그대로 재사용한다:
`M3Tokens`(필드: `surfaceContainer`, `outline`, `onSurface`, `onSurfaceVar`), `solid()`, `strokedSolid()`, `dp()`, 임포트된 `MATCH_PARENT`/`WRAP_CONTENT`/`Gravity`/`Color`.

---

## Task 1: `dropdownField()` 헬퍼 추가

**Files:**
- Modify: `Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt` (import 블록 + `chipRow()` 정의 직후)

- [ ] **Step 1: `PopupMenu` import 추가**

`MainActivity.kt`의 import 블록에서 `import android.widget.LinearLayout`(57행 부근) 바로 다음 줄에 추가한다 (알파벳 순서상 LinearLayout과 ScrollView 사이):

```kotlin
import android.widget.PopupMenu
```

- [ ] **Step 2: `dropdownField()` 함수 추가**

`chipRow()` 함수 정의가 끝나는 지점(닫는 `}` 다음, `filledButton()` 정의 앞, 약 1617~1618행)에 아래 함수를 추가한다. 시그니처는 `chipRow()`와 동일하다.

```kotlin
    private fun dropdownField(t: M3Tokens, options: List<String>, selected: String,
                             onPick: (String) -> Unit): View {
        val enabled = options.isNotEmpty()
        val field = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            background = strokedSolid(
                if (enabled) t.surfaceContainer else Color.TRANSPARENT, 8, t.outline, 1)
            setPadding(dp(12), dp(10), dp(12), dp(10))
            layoutParams = LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT)
        }
        val valueView = TextView(this).apply {
            text = if (enabled) selected else "—"
            setTextColor(if (enabled) t.onSurface else t.onSurfaceVar)
            textSize = 14f
            typeface = Typeface.DEFAULT_BOLD
            layoutParams = LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f)
        }
        val arrow = TextView(this).apply {
            text = "▾"   // ▾
            setTextColor(t.onSurfaceVar)
            textSize = 14f
        }
        field.addView(valueView)
        field.addView(arrow)
        if (enabled) {
            field.setOnClickListener { anchor ->
                val menu = PopupMenu(this, anchor)
                options.forEachIndexed { i, opt ->
                    menu.menu.add(0, i, i, opt).apply {
                        isCheckable = true
                        isChecked = opt == selected
                    }
                }
                menu.setOnMenuItemClickListener { item ->
                    onPick(options[item.itemId])
                    true
                }
                menu.show()
            }
        }
        return field
    }
```

- [ ] **Step 3: 컴파일 검증**

Run: `./Android/gradlew :SampleTestAPP:compileDebugKotlin`
Expected: `BUILD SUCCESSFUL`. (`dropdownField`는 아직 호출되지 않아 "never used" 경고가 날 수 있으나 에러는 아니다.)

- [ ] **Step 4: 커밋**

```bash
git add Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt
git commit -m "feat: add dropdownField() Material dropdown helper"
```

---

## Task 2: Run/OpenAI 탭 FAMILY를 드롭다운으로 교체

**Files:**
- Modify: `Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt:565`

- [ ] **Step 1: FAMILY 호출부 함수명 교체**

565행의 `chipRow(` 를 `dropdownField(` 로 바꾼다. 인자와 람다는 변경하지 않는다. 변경 후 블록은 다음과 같아야 한다:

```kotlin
            body.addView(labelView(t, "FAMILY"))
            body.addView(dropdownField(t, ModelCatalog.families(), selFamily) { picked ->
                selFamily = picked
                selRuntime = ModelCatalog.runtimesFor(selFamily).firstOrNull() ?: selRuntime
                selBackend = ModelCatalog.backendsFor(selFamily, selRuntime).firstOrNull() ?: selBackend
                modelPathText = defaultModelPathFor(selDescriptor, selectedQuant) ?: ""
                rebuildUi(resetModelPath = true)
            })
```

- [ ] **Step 2: 컴파일 검증**

Run: `./Android/gradlew :SampleTestAPP:compileDebugKotlin`
Expected: `BUILD SUCCESSFUL`.

- [ ] **Step 3: 커밋**

```bash
git add Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt
git commit -m "feat: use dropdownField for FAMILY in Run/OpenAI tab"
```

---

## Task 3: Chat 탭 FAMILY를 드롭다운으로 교체

**Files:**
- Modify: `Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt:796`

- [ ] **Step 1: FAMILY 호출부 함수명 교체**

796행의 `chipRow(` 를 `dropdownField(` 로 바꾼다. 인자와 람다는 변경하지 않는다. 변경 후 블록은 다음과 같아야 한다:

```kotlin
        modelCard.addView(labelView(t, "FAMILY"))
        modelCard.addView(dropdownField(t, ModelCatalog.families(), chatSelFamily) { picked ->
            chatSelFamily = picked
            chatSelRuntime = ModelCatalog.runtimesFor(chatSelFamily).firstOrNull() ?: chatSelRuntime
            chatSelBackend = ModelCatalog.backendsFor(chatSelFamily, chatSelRuntime).firstOrNull() ?: chatSelBackend
            clearChatSessionState()
            rebuildUi()
        })
```

- [ ] **Step 2: 컴파일 검증**

Run: `./Android/gradlew :SampleTestAPP:compileDebugKotlin`
Expected: `BUILD SUCCESSFUL`. `dropdownField`가 이제 사용되므로 "never used" 경고도 사라진다.

- [ ] **Step 3: 커밋**

```bash
git add Android/SampleTestAPP/src/main/java/com/example/sampletestapp/MainActivity.kt
git commit -m "feat: use dropdownField for FAMILY in Chat tab"
```

---

## Task 4: 빌드·설치 후 수동 검증

**Files:** (코드 변경 없음 — 검증 전용)

- [ ] **Step 1: 디바이스 연결 및 환경 변수 확인**

```bash
adb devices            # R3CX80H8Y0F (SM-S936U)가 'device' 상태인지 확인
echo "$JAVA_HOME $ANDROID_HOME $NDK_ROOT $ANDROID_SERIAL"   # 4개 모두 출력되는지
```
Expected: 대상 기기가 `device` 상태, 4개 변수 모두 비어있지 않음.

- [ ] **Step 2: APK 설치 (서명 충돌 시 재설치)**

먼저 빠른 설치를 시도한다 (네이티브 라이브러리는 유지됨):

```bash
./Android/gradlew ":SampleTestAPP:installDebug"
```

`INSTALL_FAILED_UPDATE_INCOMPATIBLE ... signatures do not match` 가 나오면:

```bash
adb -s "$ANDROID_SERIAL" uninstall com.example.sampletestapp
./Android/gradlew ":SampleTestAPP:installDebug"
```

Expected: `BUILD SUCCESSFUL`, 기기에 앱 설치됨.

- [ ] **Step 3: 수동 확인 (기기에서 직접)**

다음을 모두 확인한다:
1. **Run/OpenAI 탭**: FAMILY 영역이 칩 행이 아니라 드롭다운 필드(`현재값 ... ▾`)로 표시된다.
2. **Chat 탭**: FAMILY 영역이 동일하게 드롭다운 필드로 표시된다.
3. **드롭다운 동작**: FAMILY 필드를 탭하면 family 목록 팝업이 뜨고, 현재 선택값에 체크 표시가 있다.
4. **Cascading 회귀 없음**: 다른 family를 선택하면 RUNTIME/BACKEND 칩이 자동으로 재계산되어 바뀐다 (기존 칩 시절과 동일 동작). MODEL PATH도 갱신된다.
5. **다른 축 유지**: RUNTIME / BACKEND / QUANTIZATION은 여전히 칩 행으로 표시된다.
6. **테마**: 다크/라이트 모드에서 드롭다운 필드 테두리·글자색이 깨지지 않는다.

- [ ] **Step 4: 검증 결과 기록**

위 6개 항목 결과를 사용자에게 보고한다. 문제가 있으면 해당 태스크로 돌아가 수정한다. 모두 통과하면 완료.

---

## Self-Review 메모

- **스펙 커버리지**: FAMILY만(✓ Task 1~3에서 다른 축 미변경), 두 탭(✓ Task 2 Run/OpenAI, Task 3 Chat), Material 드롭다운+PopupMenu(✓ Task 1), 빈 목록 방어(✓ Task 1 `enabled` 분기), 라이트/다크(✓ 토큰 사용 + Task 4 Step 3-6), cascading 보존(✓ 람다 무변경) — 스펙 모든 요구사항이 태스크로 매핑됨.
- **타입/시그니처 일관성**: `dropdownField(t: M3Tokens, options: List<String>, selected: String, onPick: (String) -> Unit): View` — Task 1 정의와 Task 2/3 호출 인자 순서·타입 일치. `chipRow`와 동일 시그니처라 호출부는 이름만 교체.
- **플레이스홀더**: 없음 (모든 코드/명령 전체 기재).
