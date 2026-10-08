executable(
  'agxforge_mesa', 'bridge.c',
  include_directories: [inc_include, inc_src],
  dependencies: [idep_nir, idep_mesautil, idep_libagx],
  link_with: [libasahi_compiler, libagx2_disasm],
  c_args: [no_override_init_args],
  install: false,
)
