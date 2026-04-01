and_conf = configuration_data()

flags = []
if get_option('debug')
   flags += '-g'
endif


if get_option('enable-fp16')
   flags += '-march=armv8.2-a+fp16'
endif


qnn_sources = [
  meson.current_source_dir() / 'qnn_rpc_manager.cpp',
]

qnn_inc_abs = []
qnn_inc_abs += meson.current_source_dir()

qnn_inc = []
qnn_inc += include_directories('.')

# Add include paths so #include <IOTensor.hpp> works
qnn_inc += include_directories('qnn', 'qnn/Utils', 'qnn/Log', 'qnn/PAL/include')
qnn_inc_abs += meson.current_source_dir() / 'qnn'
qnn_inc_abs += meson.current_source_dir() / 'qnn' / 'Utils'
qnn_inc_abs += meson.current_source_dir() / 'qnn' / 'Log'
qnn_inc_abs += meson.current_source_dir() / 'qnn' / 'PAL' / 'include'

# Include QNN subdirectories
subdir('qnn')
subdir('qnn-api')


flags = extra_defines + flags
and_conf.set('MESON_CFLAGS',' '.join(flags))
and_conf.set('MESON_CXXFLAGS', ' ' .join(flags))
and_conf.set('MESON_QNN_CONTEXT_INC', ' '.join(qnn_inc_abs))
and_conf.set('MESON_QNN_CONTEXT_SRCS',' '.join(qnn_sources))

configure_file(input: 'Android.mk.in', output: 'Android.mk',
  configuration: and_conf
)

qnn_prefix=get_option('prefix')
qnn_libdir = qnn_prefix / get_option('libdir')

configure_file(input: 'Application.mk', output: 'Application.mk',
  copy: true
)

# below is list of file that meson acknowledges this is the output of custom target so that
# this has to be installed by meson
outputs= [
  'arm64-v8a'
]

ndk_build = find_program('ndk-build', required : true)

ndk_args = {
  'NDK_LIBS_OUT': meson.current_build_dir() / '..',
  'NDK_PROJECT_PATH': meson.current_build_dir() / '..'
}
num_threads = run_command('grep', '-c', '^processor', '/proc/cpuinfo').stdout().strip()
message('num processor are: ' + num_threads)

thread_opt_flag = '-j' + num_threads

ndk_additional_flags = [thread_opt_flag]

ndk_build_command = [ndk_build]
foreach key, val : ndk_args
  ndk_build_command += '@0@=@1@'.format(key, val)
endforeach
ndk_build_command += ndk_additional_flags

ndk_depends = []
ndk_input = []
foreach src: qnn_sources
  ndk_input += files(src)
endforeach

qnn_lib_tgt = custom_target('ndk-build',
                           input: ndk_input,
                           output: outputs,
                           build_by_default: true,
                           depends: ndk_depends,
                           install: true,
                           install_dir: qnn_libdir,
                           command: [ndk_build_command])

