#include "Optimizer/PassPipeLine.h"

#include "Optimizer/Passes.h"

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Dialect/Async/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <string>
#include <string_view>

using namespace rust_compiler::optimizer;

namespace rust_compiler::optimizer {

void createDefaultOptimizerPassPipeline(mlir::PassManager &pm,
                                        std::string_view summaryFile) {

  SummaryWriterPassOptions options;
  options.summaryOutputFile = std::string(summaryFile);

  // Hir

  pm.addPass(optimizer::createConvertHirToMirPass());
  // Mir

  pm.addPass(optimizer::createConvertMirToLirPass());
  // Lir

  pm.addPass(mlir::createSCCPPass());
  pm.addPass(optimizer::createDeadArgumentEliminationPass());
  pm.addPass(optimizer::createLoopPass());
  pm.addPass(optimizer::createFuncSpecialPass());

  pm.addPass(optimizer::createConvertLirToLLVMPass());
  // LLLVM Dialect

  // optimize
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // pm.addPass(optimizer::createCombinerPass());
  pm.addPass(mlir::createSCCPPass());

  pm.addPass(mlir::createInlinerPass());

  //  pm.addPass(mlir::createLoopInvariantCodeMotionPass());

  pm.addPass(createAttributer());
  // pm.addPass(createGVNPass());
  // pm.addPass(createRewritePass());
  // pm.addPass(createDeadCodeEliminationPass());

  pm.addPass(createSummaryWriterPass(options));

  // lower
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // pm.addPass(optimizer::createLowerErrorPropagationPass());
  // pm.addPass(optimizer::createLowerAwaitPass());
  //  pm.addPass(mlir::createAsyncFuncToAsyncRuntimePass());
  //  pm.addPass(mlir::createAsyncToAsyncRuntimePass());
  //  pm.addPass(mlir::createConvertAsyncToLLVMPass());
  // pm.addPass(createLowerUtilsToLLVMPass());

  // Finish lowering the Mir IR to the LLVM dialect.
  // pm.addPass(createMirToLLVMLowering());
}

} // namespace rust_compiler::optimizer
