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

target_link_libraries(codegen MirDialect ast ${llvm_libs} MLIRFuncDialect)
