# Extension models build configuration.
#
# CPU model subdirectories (each providing its own meson.build) are
# auto-discovered, except the `qnn` subtree which is gated on enable_qnn below.
# The public tree ships no CPU extension models; proprietary model directories
# dropped into this tree (and git-ignored) are picked up automatically.
_py = find_program('python3')
_cpu_model_dirs = run_command(
  _py, '-c',
  'import os,sys; d=sys.argv[1]; print("\\n".join(n for n in sorted(os.listdir(d)) if n != "qnn" and os.path.isfile(os.path.join(d, n, "meson.build"))))',
  meson.current_source_dir(),
  check: true,
).stdout().strip().split('\n')
foreach _d : _cpu_model_dirs
  if _d != ''
    subdir(_d)
  endif
endforeach

if enable_qnn
    subdir('qnn')
endif
