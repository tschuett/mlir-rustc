#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Dominance.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_TEST
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class TestPass : public rust_compiler::optimizer::impl::TestBase<TestPass> {
public:
  TestPass() = default;
  TestPass(const TestPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

TestPass::TestPass(const TestPass &pass)
    : rust_compiler::optimizer::impl::TestBase<TestPass>(pass) {}

llvm::StringRef TestPass::getDescription() const { return "test pass"; }

void TestPass::runOnOperation() {
  //  mlir::Operation *op = getOperation();
  //  mlir::func::FuncOp func = mlir::cast<mlir::func::FuncOp>(op);
}

std::unique_ptr<mlir::Pass> createTestPass() {
  return std::make_unique<TestPass>();
}

// https://reviews.llvm.org/D140415
