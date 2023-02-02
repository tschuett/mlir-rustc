add_library(crate_builder
           CrateBuilder.cpp
           Item.cpp
           )


target_include_directories(crate_builder PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs Remarks)

target_link_libraries(crate_builder ast hir adt ${llvm_libs})


target_precompile_headers(crate_builder PRIVATE )
