#pragma once

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

enum class FunctionQualifierKind { Const, Async, Unsafe };

class FunctionQualifiers {
  FunctionQualifierKind kind;

public:
  FunctionQualifierKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
