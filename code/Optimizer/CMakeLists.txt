add_library(optimizer
           Attributer.cpp
           Test.cpp
           Rewriter.cpp
           LowerAwait.cpp
           LowerErrorPropagation.cpp
           SummaryWriterPass.cpp
           GVN.cpp
           DeadCodeElimination.cpp
           ComputeKnownBits.cpp
           DeadStoreElimination.cpp
           SCCP.cpp
           )
#           SCCPPass.cpp


target_include_directories(optimizer PRIVATE  ../include)

add_dependencies(optimizer OptimizerPassIncGen MirOpsIncGen CombineIncGen)


target_link_libraries(optimizer analysis hir MirDialect)

target_precompile_headers(optimizer PRIVATE )



set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(Combine.cpp.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)
