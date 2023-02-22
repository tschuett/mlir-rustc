#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

enum class ArithmeticOrLogicalExpressionKind {
  Addition,
  Subtraction,
  Multiplication,
  Division,
  Remainder,
  BitwiseAnd,
  BitwiseOr,
  BitwiseXor,
  LeftShift,
  RightShift
};

class ArithmeticOrLogicalExpression final : public OperatorExpression {
  ArithmeticOrLogicalExpressionKind kind;
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  ArithmeticOrLogicalExpression(rust_compiler::Location loc)
      : OperatorExpression(
            loc, OperatorExpressionKind::ArithmeticOrLogicalExpression) {}

  void setKind(ArithmeticOrLogicalExpressionKind _kind) { kind = _kind; }
  ArithmeticOrLogicalExpressionKind getKind() const { return kind; }

  void setLhs(std::shared_ptr<Expression> _left) { left = _left; }
  void setRhs(std::shared_ptr<Expression> _right) { right = _right; }
  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };
};

} // namespace rust_compiler::ast
