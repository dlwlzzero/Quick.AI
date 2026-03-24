project('src', 'c', 'cpp',
  version: '0.1.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'werror=false',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

# ── Path configuration ──────────────────────────────────────────────────
# nntrainer submodule lives at the project root (one level up)
project_root       = meson.current_source_dir() / '..'
nntrainer_root     = project_root / 'nntrainer'
nntrainer_builddir = nntrainer_root / get_option('nntrainer_builddir')
causallm_root      = nntrainer_root / 'Applications' / 'CausalLM'

# ── Find pre-built nntrainer libraries ───────────────────────────────────
cxx = meson.get_compiler('cpp')

libnntrainer = cxx.find_library('nntrainer',
  dirs: [nntrainer_builddir / 'nntrainer'])

libccapi = cxx.find_library('ccapi-nntrainer',
  dirs: [nntrainer_builddir / 'api' / 'ccapi'])

# nntrainer includes — use compiler args for paths outside source tree
# NOTE: Do NOT include nntrainer_builddir itself — it contains no headers,
# and on systems built with enable-ruy=true, the builddir/subprojects/ruy/
# tree contains ruy/time.h which shadows the system <time.h>, causing
# "'clock' has not been declared in '::'" errors.
nntrainer_inc_args = []
foreach p : [
  nntrainer_root / 'api' / 'ccapi' / 'include',
  nntrainer_root / 'api',
  nntrainer_root / 'nntrainer',
  nntrainer_root / 'nntrainer' / 'layers',
  nntrainer_root / 'nntrainer' / 'layers' / 'loss',
  nntrainer_root / 'nntrainer' / 'models',
  nntrainer_root / 'nntrainer' / 'graph',
  nntrainer_root / 'nntrainer' / 'compiler',
  nntrainer_root / 'nntrainer' / 'optimizers',
  nntrainer_root / 'nntrainer' / 'tensor',
  nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend',
  nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'fallback',
  nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'cblas_interface',
  nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'x86',
  nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'ggml_interface',
  nntrainer_root / 'nntrainer' / 'tensor' / 'cpu_backend' / 'ggml_interface' / 'nntr_ggml_impl',
  nntrainer_root / 'nntrainer' / 'utils',
  nntrainer_root / 'nntrainer' / 'dataset',
  nntrainer_root / 'nntrainer' / 'schema',
]
  nntrainer_inc_args += ['-I' + p]
endforeach

nntrainer_dep = declare_dependency(
  dependencies: [libnntrainer, libccapi],
  compile_args: nntrainer_inc_args,
)

# ── OpenMP ───────────────────────────────────────────────────────────────
openmp_dep = dependency('openmp', required: false)

# ── CausalLM sources and includes ────────────────────────────────────────
# Explicitly list source directories to avoid pulling in build artifacts
# (e.g. tokenizers-cpp-build/) that may appear under the CausalLM tree.

# NOTE: Do NOT include causallm_root / 'api' here — the API layer
# (causal_lm_api.cpp, model_config.cpp) is provided by api.
# Including it would cause libcausallm.so to export loadModel/setOptions/etc.,
# shadowing the api implementations via symbol interposition.
causallm_auto_src = run_command('find',
  causallm_root / 'layers',
  causallm_root / 'models',
  causallm_root / 'huggingface_tokenizer.cpp',
  causallm_root / 'llm_util.cpp',
  '-name', '*.cpp',
  '-not', '-name', 'main.cpp',
  '-not', '-name', 'test_api.cpp',
  check: true).stdout().strip().split('\n')

causallm_auto_inc_dirs = run_command('find',
  causallm_root / 'layers',
  causallm_root / 'models',
  causallm_root / 'res',
  '-type', 'd',
  check: true).stdout().strip().split('\n')

# Build include args for CausalLM directories (outside source tree)
causallm_inc_args = ['-I' + causallm_root]
foreach d : causallm_auto_inc_dirs
  causallm_inc_args += ['-I' + d]
endforeach

# Tokenizer static library
tokenizer_path = causallm_root / 'lib' / 'libtokenizers_c.a'

# ── Build CausalLM as shared library ────────────────────────────────────
causallm_lib = shared_library('causallm',
    causallm_auto_src,
    dependencies: [nntrainer_dep, openmp_dep],
    cpp_args: causallm_inc_args,
    link_args: [tokenizer_path],
    install: true,
)

causallm_dep = declare_dependency(
    link_with: causallm_lib,
    compile_args: causallm_inc_args,
)

# ── Extension model sources ─────────────────────────────────────────────
quick_dot_ai_src = []
quick_dot_ai_inc = []

subdir('models')

ext_inc = include_directories('models/gauss-2.5')

# ── Build model plugin (.so for LD_PRELOAD mode) ────────────────────────
quick_dot_ai_plugin = shared_library('quick_dot_ai',
    quick_dot_ai_src,
    include_directories: [ext_inc],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep],
    install: true,
)

# ── Static lib for linking into executable (ensures constructor runs) ───
quick_dot_ai_static = static_library('quick_dot_ai_static',
    quick_dot_ai_src,
    include_directories: [ext_inc],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep],
)

# ── Build executable (original main.cpp + custom plugin) ────────────────
executable('quick_dot_ai',
    causallm_root / 'main.cpp',
    include_directories: [ext_inc],
    dependencies: [nntrainer_dep, causallm_dep, openmp_dep],
    link_whole: quick_dot_ai_static,
    cpp_args: causallm_inc_args,
    install: true,
)
