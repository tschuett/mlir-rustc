#pragma once

#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

namespace rust_compiler::analysis::attributer {

// Represents a position in the IR.
// This may directly reference an MLIR instance such as a Value or Operation or
// reference an abstract location such as a returned value from a callable.
//
// This is the MLIR equivalent to the IRPosition used in LLVM (see
// llvm/Transforms/IPO/Attributor.h).

class IRPosition {
  enum class Kind {
    ReturnedValue,
  };

public:
  static IRPosition forReturnedValue(mlir::Operation *op, unsigned resultIdx) {
    return IRPosition(Kind::ReturnedValue, op, resultIdx);
  }

  bool isReturnedValue() const { return kind == Kind::ReturnedValue; }

  void *getPosition() { return ptr; }

private:
  explicit IRPosition(Kind kind, void *ptr, unsigned ordinal)
      : kind(kind), ptr(ptr), ordinal(ordinal) {}

  Kind kind;
  void *ptr;

  [[maybe_unused]] unsigned ordinal; // used only with ReturnedValue
};

} // namespace rust_compiler::analysis::attributer
