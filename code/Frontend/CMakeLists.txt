add_library(Frontend
           FrontendAction.cpp
           FrontendActions.cpp
           CompilerInvocation.cpp
           CompilerInstance.cpp
           )

target_include_directories(Frontend PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs BitReader BitWriter)


target_link_libraries(Frontend ast ${llvm_libs})
