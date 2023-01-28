#include "Analysis/Attributer/Attributer.h"

#include "Analysis/Attributer/AAIsDead.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::analysis::attributor {

void Attributor::setup() {
  module.walk([&](mlir::Operation *op) {
    if (mlir::func::FuncOp fun = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
    } else if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    }
    //    {
    //      IRPosition FPos = IRPosition::forFuncOp(fun);
    //      getOrCreateAAFor<AAIsDead>(FPos);
    //      // for(auto& rs : fun.getResultTypes()) {
    //      //
    //      // }
    //    }
  });
}

void Attributor::recordDependence(const AbstractElement &FromAA,
                                  const AbstractElement &ToAA,
                                  DepClass DepClass) {
  assert(false);
  if (DepClass == DepClass::NONE)
    return;

  if (FromAA.getState().isAtFixpoint())
    return;

  depGraph.addEdge(&FromAA, &ToAA, DepClass);
}

Attributor::Attributor(mlir::ModuleOp module) {}

const IRPosition IRPosition::EmptyKey(Kind::Block,
                                      llvm::DenseMapInfo<void *>::getEmptyKey(),
                                      0);

const IRPosition
    IRPosition::TombstoneKey(Kind::Block,
                             llvm::DenseMapInfo<void *>::getTombstoneKey(), 0);

} // namespace rust_compiler::analysis::attributor
