add_library(codegen
           ToLLVM.cpp
           PassPipeline.cpp
           DumpLLVMIR.cpp
           LowerToLLVM.cpp
           LowerUtilsToLLVM.cpp
           TBAABuilder.cpp
           )


target_include_directories(codegen PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs TargetParser)

target_link_libraries(codegen MirDialect optimizer analysis  ast ${llvm_libs} MLIRFuncDialect MLIRAsyncDialect MLIRPass MLIRAsyncToLLVM MLIRAsyncDialect MLIRAsyncTransforms MLIRMemRefToLLVM)


add_dependencies(codegen CodeGenPassIncGen MirOpsIncGen)



target_precompile_headers(codegen PRIVATE ToLLVM.h PassPipeline.h DumpLLVMIR.h)
