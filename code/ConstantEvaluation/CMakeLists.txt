add_library(ConstantEvaluation
           ConstantEvaluation.cpp
           Owner.cpp
           )

target_include_directories(ConstantEvaluation PRIVATE  ../include)

llvm_map_components_to_libnames(llvm_libs Support)


target_link_libraries(
        ConstantEvaluation
        PRIVATE
        TyCtx
        ${llvm_libs}
        )
