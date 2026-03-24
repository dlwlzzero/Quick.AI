project('api', 'c', 'cpp',
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
project_root       = meson.current_source_dir() / '..'
nntrainer_root     = project_root / 'nntrainer'
nntrainer_builddir = nntrainer_root / get_option('nntrainer_builddir')
causallm_root      = nntrainer_root / 'Applications' / 'CausalLM'
quick_dot_ai_root  = project_root / 'src'
quick_dot_ai_build = quick_dot_ai_root / 'builddir'

# ── Find pre-built nntrainer libraries ───────────────────────────────────
cxx = meson.get_compiler('cpp')

libnntrainer = cxx.find_library('nntrainer',
  dirs: [nntrainer_builddir / 'nntrainer'])

libccapi = cxx.find_library('ccapi-nntrainer',
  dirs: [nntrainer_builddir / 'api' / 'ccapi'])

# ── Find pre-built src library ──────────────────────────────────
libcausallm = cxx.find_library('causallm',
  dirs: [quick_dot_ai_build])

# ── nntrainer includes ───────────────────────────────────────────────────
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

# ── CausalLM includes ───────────────────────────────────────────────────
causallm_auto_inc_dirs = run_command('find', causallm_root,
  '-type', 'd',
  check: true).stdout().strip().split('\n')

causallm_inc_args = []
foreach d : causallm_auto_inc_dirs
  causallm_inc_args += ['-I' + d]
endforeach

# ── Find pre-built src extension library ─────────────────────────
libquick_dot_ai = cxx.find_library('quick_dot_ai',
  dirs: [quick_dot_ai_build])

# Extension model headers
quick_dot_ai_inc_args = [
  '-I' + quick_dot_ai_root / 'models' / 'gauss-2.5',
]

# ── Dependencies ─────────────────────────────────────────────────────────
nntrainer_dep = declare_dependency(
  dependencies: [libnntrainer, libccapi],
  compile_args: nntrainer_inc_args,
)

causallm_dep = declare_dependency(
  dependencies: [libcausallm],
  compile_args: causallm_inc_args,
)

quick_dot_ai_dep = declare_dependency(
  dependencies: [libquick_dot_ai],
  compile_args: quick_dot_ai_inc_args,
)

openmp_dep = dependency('openmp', required: false)
thread_dep = dependency('threads')

# ── API include directory ────────────────────────────────────────────────
api_inc = include_directories('.')

# ── Build libquick_dot_ai_api.so ──────────────────────────────────────────
quick_dot_ai_api_lib = shared_library('quick_dot_ai_api',
    'quick_dot_ai_api.cpp',
    'model_config.cpp',
    include_directories: [api_inc],
    dependencies: [nntrainer_dep, causallm_dep, quick_dot_ai_dep, openmp_dep, thread_dep],
    install: true,
)
