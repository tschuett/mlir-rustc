add_executable(rustc-opt
           Main.cpp
           )

target_include_directories(rustc-opt PRIVATE  ../../code/include)

llvm_map_components_to_libnames(llvm_libs Support)


target_link_libraries(rustc-opt optimizer analysis ${llvm_libs} MLIROptLib )
