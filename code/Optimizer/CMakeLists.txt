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
           HirLICM.cpp
           ConvertHirToMir.cpp
           ConvertMirToLir.cpp
           ConvertLirToLLVM.cpp
           Loop.cpp
           SCCPPass.cpp
           PassPipeline.cpp
           DeadArgumentElimination.cpp
           FuncSpecialization.cpp
           LoopPlan.cpp
           MemoryAnalysis.cpp
           TestAliasAnalysisPass.cpp
           LowerUtilsToLLVM.cpp
           )

target_include_directories(optimizer PRIVATE  ../include)

add_dependencies(optimizer OptimizerPassIncGen MirOpsIncGen CombineIncGen)


target_link_libraries(optimizer
                PRIVATE
                analysis
                hir
                MirDialect
                LirDialect
                MLIRAsyncDialect
                MLIRMemRefDialect
                MLIRArithDialect
                MLIRFuncDialect
                MLIRIR
                MLIRReduce
                MLIRControlFlowToLLVM
                MLIRMemRefToLLVM
                MLIRFuncToLLVM
                MLIRVectorToLLVM
                )

target_precompile_headers(optimizer PRIVATE )



set(LLVM_TARGET_DEFINITIONS Combine.td)
mlir_tablegen(Combine.cpp.inc -gen-rewriters)
add_public_tablegen_target(CombineIncGen)
