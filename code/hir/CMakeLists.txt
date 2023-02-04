add_library(hir
           HirDialect.cpp
           HirOps.cpp
           HirTypes.cpp
           HirAttr.cpp
           HirInterfaces.cpp
           )


target_include_directories(hir PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs Remarks)

target_link_libraries(hir ${llvm_libs})



add_dependencies(hir HirOpsIncGen HirOpsIncIFaceGen)


target_precompile_headers(hir PRIVATE )
