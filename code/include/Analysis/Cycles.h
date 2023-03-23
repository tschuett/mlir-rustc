#pragma once

#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <vector>

namespace rust_compiler::analysis {

using namespace mlir;

/// Nesting of Reducible and Irreducible Loops
/// Paul Havlak, 1997
class Cycles {

public:
  void analyze(mlir::func::FuncOp *f);

private:
  /// DFS in preorder
  void depthFirstSearch(mlir::Block *block, uint32_t currentDepth);

  /// dominator tree
  mlir::DominanceInfo domInfo;

  /// current function
  mlir::func::FuncOp *fun;

  llvm::SmallMapVector<mlir::Block *, uint32_t, 8> depths;
};

} // namespace rust_compiler::analysis
