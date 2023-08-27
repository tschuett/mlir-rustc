set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(RustCOptsTableGen)

add_executable(rustc
           RustC.cpp
#           CrateBuilder.cpp
           )


target_include_directories(rustc PRIVATE  ../../code/include)

add_dependencies(rustc RustCOptsTableGen)

llvm_map_components_to_libnames(llvm_libs Option X86CodeGen X86Info X86Desc TargetParser Support MC)

target_link_libraries(rustc
        PRIVATE
        Frontend
        Serialization
        MLIRAsmParser
        ${llvm_libs})
