add_library(codegen
           CrateBuilder.cpp
           ToLLVM.cpp
           PassPipeline.cpp
           DumpLLVMIR.cpp
           LowerToLLVM.cpp
           )


target_include_directories(codegen PRIVATE  ../include)


target_link_libraries(codegen Mir)
