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


add_dependencies(codegen MirOpsIncGen)
target_link_libraries(codegen MirDialect )
