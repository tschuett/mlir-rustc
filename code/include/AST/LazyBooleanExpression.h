#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

enum LazyBooleanExpressionKind { Or, And };

class LazyBooleanExpression final : public OperatorExpression {
  LazyBooleanExpressionKind kind;
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  LazyBooleanExpression(rust_compiler::Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::LazyBooleanExpression) {
  }

  void setKind(LazyBooleanExpressionKind l) { kind = l; }

  void setLhs(std::shared_ptr<Expression> e) { left = e; }
  void setRhs(std::shared_ptr<Expression> e) { right = e; }

  LazyBooleanExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };
};

} // namespace rust_compiler::ast
