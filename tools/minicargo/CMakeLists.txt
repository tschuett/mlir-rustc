set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(MiniCargoOptsTableGen)

add_executable(minicargo
           MiniCargo.cpp
           Process.cpp
           Rustc.cpp
           )

target_include_directories(minicargo PRIVATE  ../../code/include)

set_property(TARGET minicargo PROPERTY ENABLE_EXPORTS 1)

llvm_map_components_to_libnames(llvm_libs Option Support)

# MiniCargoOptsTableGen

add_dependencies(minicargo MiniCargoOptsTableGen)

target_link_libraries(minicargo
                      PRIVATE
                      toml
                      ${llvm_libs}
                      )
