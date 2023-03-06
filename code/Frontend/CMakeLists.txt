add_library(Frontend
           FrontendAction.cpp
           FrontendActions.cpp
           CompilerInvocation.cpp
           CompilerInstance.cpp
           )

target_include_directories(Frontend PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs)


target_link_libraries(Frontend CrateLoader Optimizer sema ${llvm_libs})
