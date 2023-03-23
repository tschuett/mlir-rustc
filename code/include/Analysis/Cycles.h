#pragma once

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <vector>

namespace rust_compiler::analysis {

using namespace mlir;

/// Nesting of Reducible and Irreducible Loops
/// Paul Havlak
class Cycles {

public:
  void analyze(mlir::func::FuncOp *f);
};

} // namespace rust_compiler::analysis
