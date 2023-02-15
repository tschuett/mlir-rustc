#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

enum LazyBooleanExpressionKind {
  Or,
  And
};

class LazyBooleanExpression final : public OperatorExpression {
  LazyBooleanExpressionKind kind;
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  LazyBooleanExpression(rust_compiler::Location loc,
                        LazyBooleanExpressionKind kind,
                        std::shared_ptr<Expression> left,
                        std::shared_ptr<Expression> right)
      : OperatorExpression(loc, OperatorExpressionKind::LazyBooleanExpression),
        kind(kind), left(left), right(right) {}

  LazyBooleanExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };
};

} // namespace rust_compiler::ast
