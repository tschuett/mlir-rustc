#include "Analysis/Cycles.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/Support/LLVM.h>

using namespace llvm;

namespace rust_compiler::analysis {

static bool isZero(mlir::Attribute integer) {
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(integer))
    if (intAttr.getValue() == 0)
      return true;
  return false;
}

static bool isOne(mlir::Attribute integer) {
  if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(integer))
    if (intAttr.getValue() == 1)
      return true;
  return false;
}

void LoopInfo::analyze(mlir::func::FuncOp *f) { fun = f; }

mlir::Block *Loop::getLatch() const {
  mlir::Block *header = getHeader();
  llvm::SmallPtrSet<mlir::Block *, 8> latches;

  for (auto *block : blocks)
    if (dominanceInfo.dominates(block, header))
      latches.insert(block);

  if (latches.size() > 2)
    return nullptr;

  if (latches.size() == 0)
    return nullptr;

  return *latches.begin();
}

mlir::arith::CmpIOp *Loop::getLatchCmpInst() const {
  if (mlir::Block *latch = getLatch()) {
    if (auto branch =
            mlir::dyn_cast<mlir::cf::CondBranchOp>(latch->getTerminator())) {
      if (auto cmp = mlir::dyn_cast<mlir::arith::CmpIOp>(
              branch.getCondition().getDefiningOp())) {
        return branch.getCondition().getDefiningOp();
      }
    }
    return nullptr;
  }
}

std::optional<mlir::BlockArgument> Loop::getCanonicalInductionVariable() const {
  mlir::Block *header = getHeader();

  mlir::Block *incoming = getLoopPredecessor();
  mlir::Block *backEdge = getLatch();

  if (incoming == nullptr || backEdge == nullptr)
    return std::nullopt;

  if (auto constant =
          mlir::dyn_cast<mlir::arith::ConstantOp>(incoming->getTerminator())) {
    if (isZero(constant.getValue()))
      if (auto add =
              mlir::dyn_cast<mlir::arith::AddIOp>(backEdge->getTerminator())) {
        if (incoming->getTerminator() == add.getOperand(0).getDefiningOp())
          if (auto one = mlir::dyn_cast<mlir::arith::ConstantOp>(
                  add.getOperand(1).getDefiningOp()))
            if (isOne(one.getValue()))
              return header->getArgument(0);
      }
  }

  return std::nullopt;
}

std::vector<mlir::Block *> Loop::getExitBlocks() const {
  std::vector<mlir::Block *> exitBlocks;

  for (mlir::Block *BB : blocks)
    for (mlir::Block *succ : BB->getSuccessors())
      if (!contains(succ))
        exitBlocks.push_back(succ);

  return exitBlocks;
}

mlir::Block *Loop::getLoopPreHeader() const {
  mlir::Block *header = getHeader();

  mlir::Block *pred = header->getUniquePredecessor();
  if (pred == nullptr)
    return nullptr;
  if (contains(pred))
    return nullptr;

  return pred;
}

mlir::Block *Loop::getLoopPredecessor() const {
  mlir::Block *header = getHeader();

  mlir::Block *pred = header->getSinglePredecessor();
  if (pred == nullptr)
    return nullptr;
  if (contains(pred))
    return nullptr;

  return pred;
}

} // namespace rust_compiler::analysis
