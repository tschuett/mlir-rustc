#include "Analysis/PointsTo.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace rust_compiler::analysis {

bool PointsTo::hasMemoryEffects(mlir::Operation *op) {
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Write>())
      return true;
    if (effects.hasEffect<mlir::MemoryEffects::Read>())
      return true;
    if (effects.hasEffect<mlir::MemoryEffects::Allocate>())
      return true;
    if (effects.hasEffect<mlir::MemoryEffects::Free>())
      return true;
  } else if (op->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    return true;
  } else if (op->hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    return true;
  }

  return false;
}

void PointsTo::analyzeFunction(mlir::func::FuncOp *funcOp) {
  for (auto &bblock : funcOp->getBody()) {
    for (auto &op : bblock.getOperations()) {
      if (hasMemoryEffects(&op)) {
      }
    }
  }
}

void PointsTo::computePointsTo(mlir::ModuleOp &module) {
  module.walk([&](mlir::func::FuncOp op) { analyzeFunction(&op); });
}

} // namespace rust_compiler::analysis
