#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

enum class OperatorExpressionKind {
  BorrowExpression,
  DereferenceExpression,
  ErrorPropagationExpression,
  NegationExpression,
  ArithmeticOrLogicalExpression,
  ComparisonExpression,
  LazyBooleanExpression,
  TypeCastExpression,
  AssignmentExpression,
  CompoundAssignmentExpression
};

class OperatorExpression : public ExpressionWithoutBlock {
  OperatorExpressionKind kind;

public:
  OperatorExpression(Location loc, OperatorExpressionKind kind)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::OperatorExpression),
        kind(kind) {}

  OperatorExpressionKind getKind() const { return kind; }
};

} // namespace rust_compiler::ast
