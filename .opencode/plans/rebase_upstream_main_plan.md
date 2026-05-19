# Plan: Rebase quickdotai_api_refact onto upstream/main

## Last Updated: 2025-05-19
## Status: Planning Phase
## Priority: P0 (Blocking upstream sync)

---

## 1. Executive Summary

Merge upstream/main (70 commits ahead) with quickdotai_api_refact (156 commits ahead) using **rebase strategy**. The upstream contains fundamental engine restructuring (symbolic Tensor API, ComputeOps, ThreadManager), while our feature branch contains QuickDotAI Android API, CausalLM extensions, and multimodal features.

**Estimated Duration:** 4 hours to 2 days (depending on conflict density)  
**Risk Level:** HIGH  
**Success Criteria:**
- Linux x86 build passes: `./build.sh`
- Android build passes: `./build.sh --platform=android --enable-qnn`
- APK build/install passes: `./apk-build-install.sh`
- QuickDotAI API functionality preserved (all current features working)

---

## 2. Context

### 2.1 Repository Structure
- **Upstream**: `github.com/dlwlzzero/nntrainer` (remote: `upstream/main`)
- **Origin**: `github.sec.samsung.net/j2z0-lee/nntrainer` (remote: `origin/quickdotai_api_refact`)
- **Merge Base**: `57f8bbf8` (commit: `[script] update sciprt`)
- **Working Directory**: `/home/j2z0/Internal/Quick.AI/nntrainer`

### 2.2 Branch Divergence
- **Upstream/main**: 70 commits of engine/core improvements
- **quickdotai_api_refact**: 156 commits including 8 merge commits
- **Total file changes**: 217 files (+141,577 / -7,604 lines)
- **Estimated conflicts**: 20-50+ files

### 2.3 Key Subprojects
- `nntrainer/`: Core engine with Tensor, Layer, Model, Graph
- `nntrainer/Applications/CausalLM/`: LLM application layers
- `src/`: QuickDotAI C++ API implementation
- `api/`: QuickDotAI public API headers
- `Android/`: Android Studio project with SampleTestAPP
- `xgrammar/`: Grammar constraint engine submodule

---

## 3. Risk Analysis

### 3.1 High Risk Areas

#### A. Tensor API Restructuring (upstream commits 9159ec1c ~ e6a3f48b)
- **Impact**: Tensor construction, memory allocation, and direct access patterns completely changed
- **Affected Files**:
  - `nntrainer/tensor/tensor.h`, `tensor.cpp`
  - `nntrainer/tensor/tensor_base.h`
  - `nntrainer/tensor/memory_pool.h/cpp`
  - `nntrainer/tensor/manager.h`
  - `nntrainer/tensor/tensor_pool.h/cpp`
- **QuickDotAI Impact**: `quick_dot_ai_api.cpp`, `causal_lm_api.cpp` create Tensors directly
- **Mitigation**: Use upstream's ml::train::Tensor API. Wrap QuickDotAI-specific tensor operations.

#### B. ComputeOps Introduction (upstream commits 58b35efd ~ a6c1dc91)
- **Impact**: All compute operations routed through virtual ComputeOps interface
- **Affected Files**:
  - New: `nntrainer/tensor/cpu_backend/cpu_ops_table.cpp`
  - New: `nntrainer/compute_ops/` (hypothetical directory)
  - `nntrainer/mem_allocator.cpp` (MemoryPool alloc/dealloc changes)
- **QuickDotAI Impact**: Custom ARM NEON kernels, QNN integration
- **Mitigation**: Implement QNN-specific ComputeOps subclass or bridge layer

#### C. ThreadManager Replacement (upstream commits 909f1102 ~ 65462bb5)
- **Impact**: OpenMP and old bs_thread_pool replaced with unified thread manager
- **Affected Files**:
  - `nntrainer/utils/thread_manager.h/cpp`
  - `nntrainer/utils/thread_manager_util.h/cpp`
  - `nntrainer/utils/bs_thread_pool.h` (legacy)
  - All files using `#pragma omp parallel`
- **QuickDotAI Impact**: Possible custom thread scheduling in CausalLM
- **Mitigation**: Use upstream ThreadManager. Port custom scheduling if needed.

### 3.2 Medium Risk Areas

#### D. CausalLM Layer Changes
- **Impact**: Both branches modified core CausalLM layers
- **Affected Files**:
  - `nntrainer/Applications/CausalLM/layers/embedding_layer.cpp/h`
  - `nntrainer/Applications/CausalLM/layers/mha_core.cpp/h`
  - `nntrainer/Applications/CausalLM/layers/lm_head.cpp/h`
  - `nntrainer/Applications/CausalLM/chat_template.cpp/h`
- **Mitigation**: Careful three-way merge. Preserve QuickDotAI additions (4-bit, Gemma4, FunctionGemma).

#### E. Build System Conflicts
- **Impact**: meson.build files changed on both sides
- **Files**: Multiple `meson.build` across directories, `meson_options.txt`
- **Mitigation**: Merge options. Ensure all QuickDotAI targets still build.

### 3.3 Low Risk but Important

#### F. API Signature Preservation
- **Requirement**: `quick_dot_ai_api.h` C functions and JNI signatures must NOT change
- **Files**: `api/quick_dot_ai_api.h`, `api/causal_lm_api.h`
- **Mitigation**: Internal implementation changes only. No public API modification.

---

## 4. Execution Plan

### Phase 0: Pre-flight Safety (15 minutes)

#### Step 0.1: Create Backup Branch
```bash
cd /home/j2z0/Internal/Quick.AI/nntrainer
git branch quickdotai_api_refact-backup-20250519
git branch --set-upstream-to=origin/quickdotai_api_refact quickdotai_api_refact-backup-20250519
```

#### Step 0.2: Clean Working Tree
```bash
git stash
git clean -fdn  # Check first, then:
# git clean -fd   # if clean
git submodule update --init --recursive
```

#### Step 0.3: Baseline Build Check
Build current branch to ensure it works before changes:
```bash
# Quick x86 build
cd /home/j2z0/Internal/Quick.AI
./build.sh --platform=x86 --target=src
```

**Checkpoint**: If baseline build fails, STOP. Fix baseline first.

#### Step 0.4: Document Current State
```bash
git log --oneline quickdotai_api_refact > /tmp/feature_commits.txt
git diff --stat upstream/main...quickdotai_api_refact > /tmp/diff_stat.txt
```

### Phase 1: Rebase Initiation (Immediate but Likely Fails)

#### Step 1.1: Start Rebase
```bash
cd /home/j2z0/Internal/Quick.AI/nntrainer
git checkout quickdotai_api_refact
git rebase --onto upstream/main 57f8bbf8 quickdotai_api_refact
```

**Expected**: Rebase halts with conflicts within first 10-20 commits.

#### Step 1.2: If Immediate Failure
If `git rebase` reports too many conflicts (>20 at once), consider **Squash-Rebase Hybrid** (see Phase 6 fallback).

### Phase 2: Conflict Resolution (Iterative)

#### Strategy: Resolve in this Priority Order:

**Order 1: Build System** (`meson.build`, `meson_options.txt`)
- Resolve to include BOTH upstream dependencies AND QuickDotAI targets
- Ensure `enable-transformer`, `enable-npu`, `enable-qnn` (custom) all present

**Order 2: Core Engine** (Tensor, MemoryPool, Engine)
- Accept upstream's symbolic graph implementation
- Modify QuickDotAI API files (`quick_dot_ai_api.cpp`) to use new Tensor constructors
- Example change:
  ```cpp
  // OLD (feature branch)
  Tensor t(shape, data);
  
  // NEW (upstream)
  auto t = Tensor::Map(data, shape, stride);
  // or
  ml::train::Tensor t(shape);
  t.allocate();
  ```

**Order 3: Compute Backend** (ThreadManager, GGML interface)
- Accept upstream ThreadManager
- Check if `ggml_interface*.cpp` files have QuickDotAI-specific changes
- Port ARM NEON custom kernels to new ComputeOps interface if needed

**Order 4: CausalLM Core** (Layers, ChatTemplate)
- Three-way merge of layer implementations
- Preserve QuickDotAI additions:
  - 4-bit quantized embedding (`4bit_embedding_memfix`)
  - Gemma4 support (E2B QNN)
  - FunctionGemma (function calling)
  - Multimodal (image input)
  - Streaming API
  - KV cache manager enhancements

**Order 5: API Layer** (Public Headers)
- ABSOLUTELY NO changes to function signatures in `quick_dot_ai_api.h`
- Modify `.cpp` implementations to bridge upstream changes

#### Conflict Resolution Workflow (per commit):
```bash
# During rebase, when stopped:
git status  # See conflicts

# Edit conflicting files manually (use 3-way merge markers)
# After editing:
git add <resolved-files>

# Check if buildable at this point (highly recommended every 3-5 commits)
# For quick checks:
cd /home/j2z0/Internal/Quick.AI
./build.sh --platform=x86 --target=src

# If build fails, fix before continuing
git add -A
git rebase --continue
```

### Phase 3: Build Verification (After each significant block)

#### Checkpoint A: After Core Engine Rebased (every 5 commits)
```bash
# Test Linux compilation
cd /home/j2z0/Internal/Quick.AI
./build.sh --platform=x86 --target=src
```
**Pass Criteria**: Compilation completes without error.
**Fail Action**: Fix compilation errors before rebase --continue.

#### Checkpoint B: After All Core + API Rebased
```bash
# Test with API target
./build.sh --platform=x86 --target=api,api-test
```

#### Checkpoint C: After Complete Rebase
```bash
# Full Android build (this is critical)
./build.sh --platform=android --enable-qnn
```

#### Checkpoint D: Final Integration
```bash
# APK build and install (THE ultimate test)
./apk-build-install.sh
```

### Phase 4: Regression Testing

#### Test 1: API Compilation
- `test_api.cpp` compiles and links
- All API entry points accessible

#### Test 2: Unit Tests (if available)
```bash
cd /home/j2z0/Internal/Quick.AI/nntrainer
# Run meson tests
ninja -C builddir_x86 test
```

#### Test 3: Model Loading
- SampleTestAPP can load a model (Gauss3.8, Gemma4, Qwen3)
- Basic inference works

#### Test 4: Feature Verification Checklist
- [ ] Text generation (non-streaming)
- [ ] Streaming generation
- [ ] Multimodal (image + text)
- [ ] Chat with history
- [ ] Function calling (if applicable)
- [ ] Cancel operation during inference
- [ ] Model load/unload
- [ ] KV cache working

### Phase 5: Cleanup and History Polish

#### Step 5.1: Interactive Rebase for Clean History (optional)
```bash
# If intermediate "fix" commits are messy:
git rebase -i upstream/main
# Squash WIP commits, reword unclear messages
```

#### Step 5.2: Format Code
```bash
# Run formatter if available
# find . -name "*.cpp" -o -name "*.h" | xargs clang-format -i
```

#### Step 5.3: Final Verification
- [ ] All builds pass (Linux x86, Android arm64, APK)
- [ ] No uncommitted changes
- [ ] Branch is ahead of upstream/main by expected number of commits (~156)

### Phase 6: Delivery

#### Step 6.1: Push to Origin
```bash
# After user verification:
git push origin quickdotai_api_refact --force-with-lease
```

#### Step 6.2: Cleanup Backup
```bash
git branch -d quickdotai_api_refact-backup-20250519
```

---

## 5. Fallback Strategies

### Fallback A: Squash-Rebase Hybrid (If pure rebase is too hard)
If traditional rebase generates >50 conflicts simultaneously:

```bash
# 1. Create a fresh branch from upstream/main
git checkout -b quickdotai_api_refact-rebased upstream/main

# 2. Create a single squash commit of all feature changes
git merge --squash 57f8bbf8..quickdotai_api_refact

# 3. Manually resolve all conflicts at once (instead of 70 times)
# ... resolve ...
git commit -m "feat(quickdotai): integrate QuickDotAI API on upstream/main"

# 4. Now cherry-pick individual important commits on top, or keep as single commit
```
**Pros**: One conflict resolution pass, much faster  
**Cons**: Loses granular commit history (all squashed into one)  
**Use When**: Rebase keeps failing after 2+ hours

### Fallback B: Manual Merge + Rebase Simulation
```bash
# Perform a merge but rewrite history afterward
git checkout quickdotai_api_refact
git merge upstream/main --no-ff
# Resolve all conflicts in merge commit
# Then optionally rebase the result for linear history
```

### Fallback C: Patch-based Approach
```bash
# Export all feature changes as patches
git format-patch 57f8bbf8..quickdotai_api_refact --stdout > /tmp/feature.patch
# Apply on top of upstream/main manually, resolving file-by-file
git checkout -b quickdotai_api_refact-patched upstream/main
git am --3way /tmp/feature.patch
```

---

## 6. Key Files to Watch

### 6.1 High Conflict Probability (>90%)
- `nntrainer/meson.build`
- `nntrainer/meson_options.txt`
- `nntrainer/tensor/tensor.h`
- `nntrainer/tensor/tensor.cpp`
- `nntrainer/tensor/memory_pool.cpp`
- `nntrainer/utils/thread_manager.h`
- `nntrainer/engine.cpp`
- `nntrainer/graph/network_graph.cpp`
- `nntrainer/models/neuralnet.cpp`
- `nntrainer/layers/layer_node.cpp`

### 6.2 Medium Conflict Probability (50-90%)
- `nntrainer/layers/embedding.cpp`
- `nntrainer/layers/fc_layer.cpp`
- `nntrainer/tensor/manager.cpp`
- `nntrainer/app_context.cpp`
- `Applications/CausalLM/layers/causallm_common_properties.h`
- `Applications/CausalLM/layers/mha_core.cpp`
- `Applications/CausalLM/chat_template.cpp`
- `api/quick_dot_ai_api.cpp`

### 6.3 QuickDotAI MUST Preserve (Functionality)
- `api/quick_dot_ai_api.h` - Public API signatures
- `Android/` - All Android project files
- QNN integration code in `src/` and `qnn/`
- Function calling templates
- Multimodal input handling
- Streaming infrastructure

---

## 7. Build Commands Reference

### Linux x86 Quick Build
```bash
cd /home/j2z0/Internal/Quick.AI
./build.sh --platform=x86 --target=src,api
# Expected: < 5 minutes
```

### Linux x86 with Tests
```bash
cd /home/j2z0/Internal/Quick.AI/nntrainer
meson setup builddir_test . \
  --buildtype=release \
  -Denable-app=true \
  -Denable-test=true \
  -Denable-transformer=true
ninja -C builddir_test
ninja -C builddir_test test
```

### Android Build
```bash
cd /home/j2z0/Internal/Quick.AI
export ANDROID_NDK=/home/suyeon/Android/Sdk/ndk/27.0.12077973
./build.sh --platform=android --enable-qnn --clean
# Expected: 10-30 minutes
```

### APK Build & Install
```bash
cd /home/j2z0/Internal/Quick.AI
./apk-build-install.sh
# Expected: 5-10 minutes (depends on Gradle)
```

---

## 8. Communication Plan

### After Each Phase
- Update this plan document with actual results
- Report blockers immediately
- Seek user input if strategy change needed (Fallback A/B/C)

### Success Metrics
- All builds green
- APK installs successfully
- SampleTestAPP basic features functional

---

## 9. Notes

- **DO NOT** modify public API headers (`quick_dot_ai_api.h`) unless absolutely unavoidable
- **DO** verify ARM NEON and QNN paths build correctly (not just x86)
- **WATCH** for hidden dependencies introduced by upstream's new ComputeOps/Tensro API
- **COMMIT** frequently during conflict resolution (`git rebase --continue` after each resolved commit)
- **TEST** early and often - don't rebase 50 commits then discover a build break at commit #3

---

## Appendix: Upstream Commit Categories (by merge-tree analysis)

### A. Build & CI (5 commits)
- ci: lock Windows tokenizer Rust dependencies
- causallm: fix Windows and Android builds
- [API] update install script & default config
- [API] split API's structure and internal structure  

### B. CausalLM Features (25 commits)
- causallm: honor tokenizer chat templates
- causallm: validate Gemma embedding models
- causallm: validate Qwen2.5 embedding model
- causallm: allow q4 embedding vocab remainder
- causallm: fix Qwen2 MLP conversion order
- causallm: enable Ubuntu and Android model-level unittest
- causallm: fix UINT16 KV cache handling
- causallm: support tied Q4_0 word embeddings
- causallm: add tiny model test scaffold
- causallm: fix Qwen3 generation path
- mha_core: 5-input external KV cache mode
- KVCacheManager: standalone host-side KV cache
- causallm: add ChatTemplate (Jinja2 parser)
- Support tiny-bert, optional rotary embedding

### C. Core Engine Restructuring (35 commits) - HIGHEST RISK
- **Tensor API**: symbolic graph, Pimpl, compile-end binding
- **ComputeOps**: virtual interface, cpu_ops_table, ClComputeOps
- **QNN**: backend integration, ComputeOps wiring
- **Memory**: MemAllocator per-vendor routing

### D. Threading (8 commits)
- Introduce unified Thread Manager
- Replace bs thread pool → new thread pool
- Replace OpenMP → new thread pool
- Optimize ggml kernel
- Refactor ThreadManager

---

## Sign-off

**Author**: opencode AI  
**Date**: 2025-05-19  
**Review Required**: Yes - before execution begins  
**Next Action**: Await user approval to execute Phase 0
