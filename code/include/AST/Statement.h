#pragma once

#include "AST/AST.h"
#include "Location.h"

namespace rust_compiler::ast {

enum class StatementKind {
  ItemDeclaration,
  LetStatement,
  ExpressionStatement,
  MacroInvocationSemi
};

class Statement : public Node {
  StatementKind kind;

public:
  explicit Statement(Location location)
      : Node(location) {}

  StatementKind getKind() const { return kind; }

};

} // namespace rust_compiler::ast

// FIXME: make pure
