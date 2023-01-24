#pragma once

#include "AST/OperatorExpression.h"

namespace rust_compiler::ast {

class AssignmentExpression final : public OperatorExpression {
  std::shared_ptr<Expression> left;
  std::shared_ptr<Expression> right;

public:
  AssignmentExpression(Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::AssignmentExpression){};

  void setLeft(std::shared_ptr<Expression> left);
  void setRight(std::shared_ptr<Expression> right);

    bool containsBreakExpression() override;

  std::shared_ptr<ast::types::Type> getType() override;
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
