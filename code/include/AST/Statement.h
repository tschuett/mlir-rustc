#pragma once

#include "AST/AST.h"
#include "Location.h"

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
  explicit Statement(rust_compiler::Location location)
      : Node(location), location(location) {}

  StatementKind getKind() const { return kind; }

protected:
  rust_compiler::Location location;
};

} // namespace rust_compiler::ast

// FIXME: make pure
