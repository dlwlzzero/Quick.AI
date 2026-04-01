project('quick-dot-ai', 'c', 'cpp',
  version: '0.0.1',
  license: ['apache-2.0'],
  meson_version: '>=0.50.0',
  default_options: [
    'werror=false',
    'warning_level=1',
    'c_std=gnu89',
    'cpp_std=c++17',
    'buildtype=release'
  ]
)

add_project_arguments('-DMIN_CPP_VERSION=201402', language:['c','cpp'])

cc = meson.get_compiler('c')
cxx = meson.get_compiler('cpp')

extra_defines = []
warning_flags = [
  '-Wredundant-decls',
  '-Wwrite-strings',
  '-Wformat',
  '-Wformat-nonliteral',
  '-Wformat-security',
  '-Winit-self',
  '-Waddress',
  '-Wno-multichar',
  '-Wvla',
  '-Wpointer-arith',
  '-Wno-error=varargs',
  '-ftree-vectorize'
]

warning_c_flags = [
  '-Wmissing-declarations',
  '-Wmissing-include-dirs',
  '-Wmissing-prototypes',
  '-Wnested-externs',
  '-Waggregate-return',
  '-Wold-style-definition',
  '-Wdeclaration-after-statement',
  '-Wno-error=varargs'
]

foreach extra_arg : warning_flags
  if cc.has_argument (extra_arg)
    add_project_arguments([extra_arg], language: 'c')
  endif
  if cxx.has_argument (extra_arg)
    add_project_arguments([extra_arg], language: 'cpp')
  endif
endforeach

foreach extra_arg : warning_c_flags
  if cc.has_argument (extra_arg)
    add_project_arguments([extra_arg], language: 'c')
  endif
endforeach

#####################################################
# Set install path
quick_dot_ai_prefix = get_option('prefix')
nntrainer_internal_libdir = quick_dot_ai_prefix / get_option('libdir')
nntrainer_internal_bindir = quick_dot_ai_prefix / get_option('bindir')
nntrainer_internal_includedir = quick_dot_ai_prefix / get_option('includedir')
nntrainer_internal_confdir = get_option('sysconfdir')
application_install_dir = nntrainer_internal_bindir / 'applications'

#####################################################
# set additional flags according to options
if get_option('enable-qnn')
    message('[quick-dot-ai] Start to compile QNN API')
    extra_defines += '-DENABLE_QNN=1'
endif

message('[quick-dot-ai] extra defines are:' + ' '.join(extra_defines))
foreach defs: extra_defines
    add_project_arguments(defs, language: ['c', 'cpp'])
endforeach
#####################################################

if get_option('enable-qnn')
    subdir('qnn/jni')
endif