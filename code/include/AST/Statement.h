#pragma once

#include "AST/AST.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

enum class StatementKind {
  Item,
  LetStatement,
  ExpressionStatement,
  MacroInvocationSemi
};

class Statement : public Node {
  StatementKind kind;

public:
  explicit Statement(mlir::Location location) : location(location) {}

  StatementKind getKind() const { return kind; }

protected:
  mlir::Location location;
};

} // namespace rust_compiler::ast

// FIXME: make pure
