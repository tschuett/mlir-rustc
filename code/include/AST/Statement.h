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
  explicit Statement(Location location, StatementKind kind)
      : Node(location), kind(kind) {}

  StatementKind getKind() const { return kind; }

  bool containsBreakExpression();
};

} // namespace rust_compiler::ast

// FIXME: make pure
