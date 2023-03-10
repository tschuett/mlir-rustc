#include "PassPipeline.h"

#include "CodeGen/Passes.h"
#include "Optimizer/Passes.h"

#include <mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h>
#include <mlir/Dialect/Async/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace rust_compiler::optimizer;
using namespace rust_compiler::codegen;

namespace rust_compiler {

void createDefaultOptimizerPassPipeline(mlir::PassManager &context) {
  // Hir

  pm.addPass(optimizer::createConvertHirToMirPass());
  // Mir

  pm.addPass(optimizer::createConvertMirToLirPass());
  // Lir

  pm.addPass(optimizer::createLoopPass());

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

int processMLIR(mlir::MLIRContext &context,
                mlir::OwningOpRef<mlir::ModuleOp> &module) {

  SummaryWriterPassOptions options;
  options.summaryOutputFile = "";

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  // Hir

  pm.addPass(optimizer::createConvertHirToMirPass());
  // Mir

  pm.addPass(optimizer::createConvertMirToLirPass());
  // Lir

  pm.addPass(optimizer::createLoopPass());

  pm.addPass(optimizer::createConvertLirToLLVMPass());
  // LLLVM Dialect

  // optimize
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  // pm.addPass(optimizer::createCombinerPass());
  pm.addPass(mlir::createSCCPPass());

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

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

} // namespace rust_compiler
