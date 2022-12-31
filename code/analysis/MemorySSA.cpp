#include "Analysis/MemorySSA/MemorySSA.h"

#include "Analysis/MemorySSA/MemorySSAWalker.h"

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
  }
  else if (auto store = mlir::dyn_cast<mlir::memref::StoreOp>(a))
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

static auto hasMemEffect(mlir::Operation &op) {
  struct Result {
    bool read = false;
    bool write = false;
  };

  Result ret;
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Write>())
      ret.write = true;

    if (effects.hasEffect<mlir::MemoryEffects::Read>())
      ret.read = true;
  } else if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    ret.write = true;
  }

  return ret;
}

void MemorySSA::analyzeFunction(mlir::Operation *funcOp) {
  if (auto fun = mlir::dyn_cast<mlir::FunctionOpInterface>(funcOp)) {
    for (auto &bblock : fun.getBlocks()) {
      for (auto &op : bblock.getOperations()) {
        /*auto mem =*/hasMemEffect(op);
      }
    }
  }
}

void MemorySSA::findFunctionOps() {
  mlir::Region &body = module.getBodyRegion();

  for (auto &bblock : body.getBlocks()) {
    for (auto &op : bblock.getOperations()) {
      if (isFunction(op)) {
        // functionOps.push_back(op);
      }
    }
  }
}

MemorySSAWalker *MemorySSA::buildMemorySSA() {
  if (Walker)
    return Walker;

  findFunctionOps();

  for (mlir::Operation *funcOp : functionOps) {
    analyzeFunction(funcOp);
  }

  Walker = new MemorySSAWalker(this);

  return Walker;
}

} // namespace rust_compiler::analysis
