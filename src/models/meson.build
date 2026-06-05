# Extension models build configuration
# Add subdirectories for each custom model here
fs = import('fs')
if fs.is_dir('gauss-2.5')
  subdir('gauss-2.5')
endif
#subdir('gauss-3.8-qnn')
if enable_qnn
    subdir('qnn')
endif
if fs.is_dir('ouro')
  subdir('ouro')
endif
# gauss-3 (CPU Gauss3CausalLM) disabled: still on the old layer-list Transformer
# interface and not registered in quick_dot_ai_api (only the gauss-3 *QNN*
# variants are registered). Orphaned by the symbolic-graph merge; excluded
# pending a port. The gauss-3 QNN models above are unaffected.
# subdir('gauss-3')
