add_library(codegen
           ModuleBuilder.cpp
           ToLLVM.cpp
           PassPipeline.cpp
           DumpLLVMIR.cpp
           LowerToLLVM.cpp
           TypeBuilder.cpp
           Target.cpp
           )


target_include_directories(codegen PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs)


add_dependencies(codegen MirOpsIncGen)

target_link_libraries(codegen MirDialect ${llvm_libs} MLIRFuncDialect)
