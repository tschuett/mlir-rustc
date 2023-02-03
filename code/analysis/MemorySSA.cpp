#include "Analysis/MemorySSA/MemorySSA.h"

#include "Analysis/MemorySSA/MemorySSAWalker.h"
#include "mlir/IR/Operation.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

namespace rust_compiler::analysis {

/// a and b must be memrefs
std::optional<mlir::AliasResult> MemorySSA::mayAlias(mlir::Operation *a,
                                                     mlir::Operation *b) {
  mlir::Value valueA;
  mlir::Value valueB;
  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(a)) {
    valueA = load.getMemRef();
  } else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(a))
    valueA = store.getMemRef();
  else
    return std::nullopt;

  if (auto load = mlir::dyn_cast<mlir::memref::LoadOp>(b))
    valueB = load.getMemRef();
  else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(b))
    valueB = store.getMemRef();
  else
    return std::nullopt;

  mlir::AliasResult res = aliasAnalysis->alias(valueA, valueB);

  return res;
}

bool MemorySSA::isFunction(mlir::Operation &op) {
  if (auto effects = mlir::dyn_cast<mlir::FunctionOpInterface>(op)) {
    return true;
  }
  return false;
}

//static auto getMemoryEffect(mlir::Operation &op) {
//  struct Result {
//    bool read = false;
//    bool write = false;
//  };
//
//  Result ret;
//  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
//    if (effects.hasEffect<mlir::MemoryEffects::Write>())
//      ret.write = true;
//
//    if (effects.hasEffect<mlir::MemoryEffects::Read>())
//      ret.read = true;
//  } else if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
//    ret.write = true;
//  }
//
//  return ret;
//}

static bool hasMemoryEffects(mlir::Operation &op) {
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Write>())
      return true;

    if (effects.hasEffect<mlir::MemoryEffects::Read>())
      return true;
  } else if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    return true;
  }

  return false;
}

void MemorySSA::analyzeFunction(mlir::func::FuncOp *funcOp) {
  for (auto &bblock : funcOp->getBody()) {
    for (auto &op : bblock.getOperations()) {
      if (hasMemoryEffects(op)) {
      }
    }
  }
}

MemorySSAWalker *MemorySSA::buildMemorySSA() {
  if (Walker)
    return Walker;

  module.walk([&](mlir::func::FuncOp op) {
    analyzeFunction(&op);
  });

  // findFunctionOps();

  //  for (mlir::Operation *funcOp : functionOps) {
  //    analyzeFunction(funcOp);
  //  }

  Walker = new MemorySSAWalker(this);

  return Walker;
}

} // namespace rust_compiler::analysis

// https://github.com/intel/mlir-extensions/blob/2a6d65137105e869c70fd1d86ba3bb784f70f6df/mlir/lib/analysis/memory_ssa.cpp

// LoopLikeOpInterface
