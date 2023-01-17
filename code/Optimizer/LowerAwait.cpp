#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_LOWERAWAITPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class LowerAwaitPass
    : public rust_compiler::optimizer::impl::LowerAwaitPassBase<
          LowerAwaitPass> {
public:
  LowerAwaitPass() = default;
  LowerAwaitPass(const LowerAwaitPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

LowerAwaitPass::LowerAwaitPass(const LowerAwaitPass &pass)
    : rust_compiler::optimizer::impl::LowerAwaitPassBase<LowerAwaitPass>(pass) {
}

llvm::StringRef LowerAwaitPass::getDescription() const { return "test pass"; }

void LowerAwaitPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  module.walk([&](mlir::func::FuncOp f) {
  // isAsync() -> rewrite
   // if (isa<rust_compiler::Mir::AwaitOp>(op)) {
   // }
  });
}

std::unique_ptr<mlir::Pass> createLowerAwaitPass() {
  return std::make_unique<LowerAwaitPass>();
}

// https://mlir.llvm.org/docs/Dialects/AsyncDialect/

// ::mlir::async::FuncOp
