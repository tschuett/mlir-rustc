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

class ArithmeticOrLogicalExpression : public OperatorExpression {
  ArithmeticOrLogicalExpressionKind kind;
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  ArithmeticOrLogicalExpression(rust_compiler::Location loc,
                                ArithmeticOrLogicalExpressionKind kind,
                                std::shared_ptr<Expression> left,
                                std::shared_ptr<Expression> right)
      : OperatorExpression(
            loc, OperatorExpressionKind::ArithmeticOrLogicalExpression),
        kind(kind), left(left), right(right) {}

  ArithmeticOrLogicalExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };

  size_t getTokens() override {
    return left->getTokens() + 1 + right->getTokens();
  }
};

} // namespace rust_compiler::ast
