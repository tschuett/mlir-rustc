add_library(remarks
           OptimizationRemarkEmitter.cpp
           OptimizationRemark.cpp
           )


target_include_directories(remarks PRIVATE  ../include)


llvm_map_components_to_libnames(llvm_libs Remarks)

target_link_libraries(remarks ast ${llvm_libs})


target_precompile_headers(remarks PRIVATE )
