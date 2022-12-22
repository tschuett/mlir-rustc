#include "Mir/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
struct MirToLLVMLoweringPass
    : public PassWrapper<MirToLLVMLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MirToLLVMLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MirToLLVMLoweringPass::runOnOperation() {
  // FIXME
}

namespace rust_compiler {

/// Create a pass for lowering operations the remaining `Toy` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> createLowerToLLVMPass() {
  return std::make_unique<MirToLLVMLoweringPass>();
}

} // namespace rust_compiler
