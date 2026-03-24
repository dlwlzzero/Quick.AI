project('api-app', 'c', 'cpp',
  version: '0.1.0',
  license: ['apache-2.0'],
  meson_version: '>=0.55.0',
  default_options: [
    'werror=false',
    'warning_level=1',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

# ── Path configuration ──────────────────────────────────────────────────
project_root     = meson.current_source_dir() / '..'
api_builddir     = project_root / 'api' / 'builddir'

# ── Find pre-built API library ───────────────────────────────────────────
cxx = meson.get_compiler('cpp')

libquick_dot_ai_api = cxx.find_library('quick_dot_ai_api',
  dirs: [api_builddir])

# ── API headers (copied to include/ by build script) ─────────────────────
api_inc = include_directories('include')

api_dep = declare_dependency(
  dependencies: [libquick_dot_ai_api],
  include_directories: api_inc,
)

# ── Build test executable ────────────────────────────────────────────────
executable('quick_dot_ai_test',
    'test_api.cpp',
    dependencies: [api_dep],
    install: true,
)
