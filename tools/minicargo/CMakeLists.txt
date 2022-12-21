set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(MiniCargoOptsTableGen)

add_executable(minicargo
           MiniCargo.cpp
           )

add_library(minicargolib)

target_include_directories(minicargo PRIVATE  ../../code/include)

set_property(TARGET minicargo PROPERTY ENABLE_EXPORTS 1)

target_link_libraries(minicargo STATIC MiniCargoOptsTableGen toml)
