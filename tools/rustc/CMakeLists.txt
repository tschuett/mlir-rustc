set(LLVM_TARGET_DEFINITIONS Opts.td)
tablegen(LLVM Opts.inc -gen-opt-parser-defs)
add_public_tablegen_target(RustCOptsTableGen)

add_executable(rustc
           RustC.cpp
           CrateBuilder.cpp
           )


target_include_directories(rustc PRIVATE  ../../code/include)

add_dependencies(rustc RustCOptsTableGen)

llvm_map_components_to_libnames(llvm_libs Option AArch64CodeGen AArch64Desc AArch64Info ARMCodeGen ARMDesc ARMInfo X86CodeGen X86Info X86Desc TargetParser Support MC)

target_link_libraries(rustc lexer parser sema modulebuilder MirDialect codegen MLIRAsmParser ${llvm_libs})

#target_link_libraries(rustc lexer parser  ${llvm_libs})

#target_precompile_headers(rustc PRIVATE ToLLVM.h PassPipeline.h DumpLLVMIR.h)


set_target_properties(rustc PROPERTIES
        CXX_EXCEPTIONS OFF
        CXX_RTTI OFF
    )
