add_library(hir
           Crate.cpp
           )


target_include_directories(hir PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs Remarks)

target_link_libraries(hir ${llvm_libs})


target_precompile_headers(hir PRIVATE )
