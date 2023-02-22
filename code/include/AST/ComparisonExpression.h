#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

enum class ComparisonExpressionKind {
  Equal,
  NotEqual,
  GreaterThan,
  LessThan,
  GreaterThanOrEqualTo,
  LessThanOrEqualTo
};

class ComparisonExpression final : public OperatorExpression {
  ComparisonExpressionKind kind;
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  ComparisonExpression(rust_compiler::Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::ComparisonExpression) {}

  void setKind(ComparisonExpressionKind k) { kind = k; }
  void setLhs(std::shared_ptr<Expression> e) { left = e; }
  void setRhs(std::shared_ptr<Expression> e) { right = e; }

  ComparisonExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };
};

} // namespace rust_compiler::ast
