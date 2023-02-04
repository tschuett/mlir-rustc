#include "Analysis/MemorySSA/MemorySSA.h"

#include "Analysis/MemorySSA/MemorySSAWalker.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/CallInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/LoopLikeInterface.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

namespace rust_compiler::analysis {

std::shared_ptr<Node> MemorySSA::createDef(mlir::Operation *op,
                                           std::shared_ptr<Node> arg) {
  //  return std::make_shared<Node>(op, NodeType::Def, arg);
}

std::shared_ptr<Node> MemorySSA::createUse(mlir::Operation *op,
                                           std::shared_ptr<Node> arg) {
  //return std::make_shared<Node>(op, NodeType::Use, arg);
}

std::shared_ptr<Node>
MemorySSA::createPhi(mlir::Operation *op,
                     llvm::ArrayRef<std::shared_ptr<Node>> args) {
  //  return std::make_shared<Node>(op, NodeType::Phi, args);
}

std::shared_ptr<Node> MemorySSA::getRoot() {
  //  if (root)
  //    return root;
  //root = std::make_shared<Node>(nullptr, NodeType::Root, std::nullopt);
  //nodes.push_back(root);
  return root;
}

std::shared_ptr<Node> MemorySSA::getTerm() {
  // if (term)
  //   return term;
  //term = std::make_shared<Node>(nullptr, NodeType::Term, std::nullopt);
  //nodes.push_back(term);
  return term;
}

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

bool MemorySSA::hasMemoryWriteEffect(mlir::Operation &op) {
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Write>())
      return true;
  } else if (op.hasTrait<mlir::OpTrait::HasRecursiveMemoryEffects>()) {
    return true;
  }

  return false;
}

bool MemorySSA::hasMemoryReadEffect(mlir::Operation &op) {
  if (auto effects = mlir::dyn_cast<mlir::MemoryEffectOpInterface>(op)) {
    if (effects.hasEffect<mlir::MemoryEffects::Read>())
      return true;
  }
  return false;
}

bool MemorySSA::hasMemoryEffects(mlir::Operation &op) {
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

bool MemorySSA::hasCallEffects(mlir::Operation &op) {
  if (auto callOp = mlir::dyn_cast<mlir::CallOpInterface>(op))
    return true;
  return false;
}

void MemorySSA::analyzeFunction(mlir::func::FuncOp *funcOp) {
  auto currentNode = getRoot();

  for (auto &bblock : funcOp->getBody()) {
    for (auto &op : bblock.getOperations()) {
      if (hasMemoryEffects(op)) {
        // FIXME
        // FIXME: createPhi(
        if (hasMemoryWriteEffect(op)) {
          auto newNode = createDef(&op, currentNode);
          newNode->setDominator(currentNode);
          currentNode->setPostDominator(newNode);
          currentNode = newNode;
        }
        if (hasMemoryReadEffect(op)) {
          auto node = createUse(&op, currentNode);
        }
      }
    }
  }
  auto term = getTerm();
  //FIMXE term->setArgument(0, last);
}

MemorySSAWalker *MemorySSA::buildMemorySSA() {
  if (Walker)
    return Walker;

  module.walk([&](mlir::func::FuncOp op) { analyzeFunction(&op); });

  Walker = new MemorySSAWalker(this);

  return Walker;
}

} // namespace rust_compiler::analysis

// https://github.com/intel/mlir-extensions/blob/2a6d65137105e869c70fd1d86ba3bb784f70f6df/mlir/lib/analysis/memory_ssa.cpp

/// Return the modify-reference behavior of `op` on `location`.
//  ModRefResult getModRef(Operation *op, Value location);
