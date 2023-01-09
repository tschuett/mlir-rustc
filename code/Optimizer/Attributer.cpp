#include "Optimizer/Passes.h"
#include "Analysis/Attributer/Attributer.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_TEST
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
  class AttributerPass : public rust_compiler::optimizer::impl::TestBase<AttributerPass> {
public:
  AttributerPass() = default;
  AttributerPass(const AttributerPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

AttributerPass::AttributerPass(const AttributerPass &pass)
  : rust_compiler::optimizer::impl::TestBase<AttributerPass>(pass) {}

llvm::StringRef AttributerPass::getDescription() const { return "test pass"; }

void AttributerPass::runOnOperation() {
//  mlir::Operation *op = getOperation();
//  mlir::func::FuncOp func = mlir::cast<mlir::func::FuncOp>(op);
}

std::unique_ptr<mlir::Pass> createAttributerPass() {
  return std::make_unique<AttributerPass>();
}


// https://reviews.llvm.org/D140415
