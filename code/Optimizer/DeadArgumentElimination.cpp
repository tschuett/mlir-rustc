#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_DEADARGUMENTELIMINATIONPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class DeadArgumentEliminationPass
    : public rust_compiler::optimizer::impl::DeadArgumentEliminationPassBase<
          DeadArgumentEliminationPass> {
public:
  void runOnOperation() override;
};

} // namespace

void DeadArgumentEliminationPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
  module.walk([&](mlir::func::FuncOp f) {

    
  });
}

// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/IPO/DeadArgumentElimination.cpp
