#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Operation.h>

#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_LOWERERRORPROPAGATIONPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class LowerErrorPropagationPass
    : public rust_compiler::optimizer::impl::LowerErrorPropagationPassBase<
          LowerErrorPropagationPass> {
public:
  LowerErrorPropagationPass() = default;
  LowerErrorPropagationPass(const LowerErrorPropagationPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

LowerErrorPropagationPass::LowerErrorPropagationPass(const LowerErrorPropagationPass &pass)
    : rust_compiler::optimizer::impl::LowerErrorPropagationPassBase<LowerErrorPropagationPass>(pass) {
}

llvm::StringRef LowerErrorPropagationPass::getDescription() const { return "test pass"; }

void LowerErrorPropagationPass::runOnOperation() {
  mlir::func::FuncOp f = getOperation();
  f.walk([&](mlir::Operation *op) {
    if (isa<rust_compiler::Mir::AwaitOp>(op)) {
    }
  });
}

std::unique_ptr<mlir::Pass> createLowerErrorPropagationPass() {
  return std::make_unique<LowerErrorPropagationPass>();
}
