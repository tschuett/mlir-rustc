add_library(Serialization
           ASTWriter.cpp
           ASTReader.cpp
           )

target_include_directories(Serialization PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs BitReader BitWriter)


target_link_libraries(Serialization ast ${llvm_libs})
