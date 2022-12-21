add_library(codegen
           CodeGen.cpp
           )


target_include_directories(codegen PRIVATE  ../include)


target_link_libraries(codegen Mir)
