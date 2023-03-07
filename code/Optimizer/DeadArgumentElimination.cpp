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

private:
  void surveyFunction(mlir::func::FuncOp &f);
  bool removeDeadStuffFromFunction(mlir::func::FuncOp &f);
  bool removeDeadArgumentsFromCallers(mlir::func::FuncOp &f);
};

} // namespace

void DeadArgumentEliminationPass::surveyFunction(mlir::func::FuncOp &f) {}

bool DeadArgumentEliminationPass::removeDeadStuffFromFunction(
    mlir::func::FuncOp &f) {}

bool DeadArgumentEliminationPass::removeDeadArgumentsFromCallers(
    mlir::func::FuncOp &f) {}

void DeadArgumentEliminationPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  bool changed = false;

  // Second phase: Loop through the module, determining which arguments are
  // live. We assume all arguments are dead unless proven otherwise (allowing us
  // to determine that dead arguments passed into recursive functions are dead).
  module.walk([&](mlir::func::FuncOp f) { surveyFunction(f); });

  // Now, remove all dead arguments and return values from each function in
  // turn.  We use make_early_inc_range here because functions will probably get
  // removed (i.e. replaced by new ones).
  module.walk(
      [&](mlir::func::FuncOp f) { changed |= removeDeadStuffFromFunction(f); });

  // Finally, look for any unused parameters in functions with non-local
  // linkage and replace the passed in parameters with poison.
  module.walk([&](mlir::func::FuncOp f) {
    changed |= removeDeadArgumentsFromCallers(f);
  });
}

// https://github.com/llvm/llvm-project/blob/main/llvm/lib/Transforms/IPO/DeadArgumentElimination.cpp

// FIXME varargs?
