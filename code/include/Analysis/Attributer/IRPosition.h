#pragma once

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace rust_compiler::analysis::attributor {

// Represents a position in the IR.
// This may directly reference an MLIR instance such as a Value or Operation or
// reference an abstract location such as a returned value from a callable.
//
// This is the MLIR equivalent to the IRPosition used in LLVM (see
// llvm/Transforms/IPO/Attributor.h).

class IRPosition {
  enum class Kind { ReturnedValue, Function, Block };

public:
  static const IRPosition EmptyKey;
  static const IRPosition TombstoneKey;

  static IRPosition forReturnedValue(mlir::Operation *op, unsigned resultIdx) {
    return IRPosition(Kind::ReturnedValue, op, resultIdx);
  }

  static IRPosition forFuncOp(mlir::func::FuncOp *op) {
    return IRPosition(Kind::Function, op, 0);
  }

  bool isReturnedValue() const { return kind == Kind::ReturnedValue; }

  void *getPosition() { return ptr; }

  // Conversion into a void * to allow reuse of pointer hashing.
  operator void *() const { return ptr; }

  void print(llvm::raw_ostream &os) const;
  void print(llvm::raw_ostream &os, mlir::AsmState &asmState) const;

private:
  template <typename T, typename Enable> friend struct llvm::DenseMapInfo;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &os, IRPosition pos);

  explicit IRPosition(Kind kind, void *ptr, unsigned ordinal)
      : kind(kind), ptr(ptr), ordinal(ordinal) {}

  Kind kind;
  void *ptr;

  [[maybe_unused]] unsigned ordinal; // used only with ReturnedValue
};

} // namespace rust_compiler::analysis::attributer

namespace llvm {

using rust_compiler::analysis::attributor::IRPosition;

// Helper that allows Position as a key in a DenseMap.
template <> struct DenseMapInfo<IRPosition> {
  static inline IRPosition getEmptyKey() { return IRPosition::EmptyKey; }
  static inline IRPosition getTombstoneKey() {
    return IRPosition::TombstoneKey;
  }
  static unsigned getHashValue(const IRPosition &pos) {
    return (DenseMapInfo<void *>::getHashValue(pos) << 4) ^
           (DenseMapInfo<unsigned>::getHashValue(pos.ordinal));
  }

  static bool isEqual(const IRPosition &a, const IRPosition &b) {
    return a == b;
  }
};

} // end namespace llvm
