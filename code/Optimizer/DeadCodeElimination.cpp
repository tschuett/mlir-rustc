#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_DEADCODEELIMINATIONPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class DeadCodeEliminationPass
    : public rust_compiler::optimizer::impl::DeadCodeEliminationPassBase<
          DeadCodeEliminationPass> {
public:
  void runOnOperation() override;
};

} // namespace

void DeadCodeEliminationPass::runOnOperation() {
  //  mlir::func::FuncOp f = getOperation();
  // module.walk([&](mlir::func::FuncOp f) {
  //  isAsync() -> rewrite
  //  if (isa<rust_compiler::Mir::AwaitOp>(op)) {
  //  }
  //});
}
