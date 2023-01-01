set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(RustCOptsTableGen)

add_executable(rustc
           RustC.cpp
           CrateBuilder.cpp
           )


target_include_directories(rustc PRIVATE  ../../code/include)

add_dependencies(rustc RustCOptsTableGen)

llvm_map_components_to_libnames(llvm_libs Option AArch64CodeGen ARMCodeGen X86CodeGen TargetParser)

target_link_libraries(rustc lexer parser sema MirDialect codegen MLIRAsmParser ${llvm_libs})

#target_precompile_headers(rustc PRIVATE ToLLVM.h PassPipeline.h DumpLLVMIR.h)
