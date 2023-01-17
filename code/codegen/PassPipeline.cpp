#include "PassPipeline.h"

#include "Optimizer/Passes.h"

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

using namespace rust_compiler::optimizer;

namespace rust_compiler {

int processMLIR(mlir::MLIRContext &context,
                mlir::OwningOpRef<mlir::ModuleOp> &module) {

  mlir::PassManager pm(&context);
  // Apply any generic pass manager command line options and run the pipeline.
  applyPassManagerCLOptions(pm);

  pm.addPass(optimizer::createRewriterPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(optimizer::createLowerErrorPropagationPass());
  pm.addPass(optimizer::createLowerAwaitPass());
  pm.addPass(optimizer::createAttributerPass());

  // Finish lowering the toy IR to the LLVM dialect.
  pm.addPass(createLowerToLLVMPass());

  if (mlir::failed(pm.run(*module)))
    return 4;
  return 0;
}

} // namespace rust_compiler
