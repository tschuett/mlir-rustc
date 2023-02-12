#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

#include <memory>

namespace rust_compiler::ast {

class DereferenceExpression final : public OperatorExpression {
  std::shared_ptr<Expression> right;

public:
  DereferenceExpression(rust_compiler::Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::DereferenceExpression) {
  }

  std::shared_ptr<Expression> getRHS() const { return right;};

  size_t getTokens() override;

  bool containsBreakExpression() override;
};

} // namespace rust_compiler::ast
