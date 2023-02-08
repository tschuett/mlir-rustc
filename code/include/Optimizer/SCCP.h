#pragma once

#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/StringSet.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Region.h>
#include <mlir/IR/Value.h>

namespace rust_compiler::optimizer {

/// Constant Propagation with Conditional Branches
/// MARK N. WEGMAN and F. KENNETH ZADECK

class LatticeValue {
  enum class LatticeValueKind { Bottom, Constant, Up };

public:
  void meet(const LatticeValue &rhs);

private:
  LatticeValueKind kind = LatticeValueKind::Up;
  mlir::Attribute value;
  mlir::Dialect *constantProvider;
};

class Solver {

public:
  Solver(mlir::ModuleOp &module);

  void run();

private:
  void visitCallIndirectOp(mlir::func::CallIndirectOp *op);
  void visitCallOp(mlir::func::CallOp *op);
  void visitOperation(mlir::Operation *op);
  void visitBlock(mlir::Block *block);
  void visitBlockArgument(mlir::Block *block, unsigned idx);
  void visitRegion(mlir::Region *reg);

  /// Returns true if the given block is executable.
  bool isBlockExecutable(mlir::Block *block) const;

  bool markBlockExecutable(mlir::Block *block);

  void markOverdefined(mlir::Value value);

  /// Mark all of the given values as overdefined.
  template <typename ValuesT> void markAllOverdefined(ValuesT values) {
    for (auto value : values)
      markOverdefined(value);
  }

  llvm::SmallPtrSet<mlir::Block *, 16> executableBlocks;

  /// The lattice of each SSA value.
  llvm::DenseMap<mlir::Value *, LatticeValue> latteiceValue;

  /// A worklist containing blocks that need to be processed.
  llvm::SmallVector<mlir::Block *, 64> blockWorklist;

  /// A worklist of operations that need to be processed.
  llvm::SmallVector<mlir::Operation *, 64> opWorklist;

  llvm::StringSet<> localFunctions;
};

} // namespace rust_compiler::optimizer


// FIXME overdefined terminology

// bool isConstant(mlir::Value *)

