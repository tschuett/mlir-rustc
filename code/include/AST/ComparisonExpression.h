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
  ComparisonExpression(rust_compiler::Location loc,
                       ComparisonExpressionKind kind,
                       std::shared_ptr<Expression> left,
                       std::shared_ptr<Expression> right)
      : OperatorExpression(loc, OperatorExpressionKind::ComparisonExpression),
        kind(kind), left(left), right(right) {}

  ComparisonExpressionKind getKind() const { return kind; }

  std::shared_ptr<Expression> getRHS() const { return right; };

  std::shared_ptr<Expression> getLHS() const { return left; };

  bool containsBreakExpression() override;

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
