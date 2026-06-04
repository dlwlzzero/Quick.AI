qnn_src = [
    meson.current_source_dir() / 'android_memory_allocator.cpp',
    meson.current_source_dir() / 'quick_dot_ai_qnn.cpp',
    meson.current_source_dir() / 'graph_parser.cpp',
    meson.current_source_dir() / 'generate_qnn_utils.cpp',
    meson.current_source_dir() / 'qnn_kv_cache_manager.cpp',
]

qnn_inc = include_directories('.', '../../../api')

quick_dot_ai_src += qnn_src
quick_dot_ai_inc += qnn_inc

# Auto-discover QNN model subdirectories (each provides its own meson.build).
# The public tree ships only gemma4-e2b-qnn; any additional proprietary model
# directory dropped into this tree (and git-ignored) is picked up automatically
# with no per-model edits here.
_py = find_program('python3')
_qnn_model_dirs = run_command(
  _py, '-c',
  'import os,sys; d=sys.argv[1]; print("\\n".join(n for n in sorted(os.listdir(d)) if os.path.isfile(os.path.join(d, n, "meson.build"))))',
  meson.current_source_dir(),
  check: true,
).stdout().strip().split('\n')
foreach _d : _qnn_model_dirs
  if _d != ''
    subdir(_d)
  endif
endforeach
