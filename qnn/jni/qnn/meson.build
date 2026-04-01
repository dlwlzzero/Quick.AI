qnn_inc += include_directories('.')
qnn_inc += include_directories('./QNN', './Log', './PAL/include', './Utils', './WrapperUtils')
qnn_inc_abs += meson.current_source_dir() / 'QNN'
qnn_inc_abs += meson.current_source_dir()

qnn_elem = [
  'Log',
  'op',
  'PAL',
  'Utils',
  'WrapperUtils'
]

qnn_headers = []

qnn_headers_ =[
  'QNN.hpp',
  'QnnTypeMacros.hpp',
  'qnn_properties.h'
]

qnn_sources += meson.current_source_dir() / 'qnn_properties.cpp'

foreach e : qnn_elem
  subdir(e)
  qnn_inc += include_directories(e)
  qnn_inc_abs += meson.current_source_dir() / e
endforeach

  qnn_inc_abs += meson.current_source_dir() / 'PAL' / 'include'

