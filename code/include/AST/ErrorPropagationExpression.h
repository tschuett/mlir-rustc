#pragma once

#include "AST/Expression.h"
#include "AST/OperatorExpression.h"

namespace rust_compiler::ast {

class ErrorPropagationExpression : public OperatorExpression {
  std::shared_ptr<Expression> expr;

public:
  ErrorPropagationExpression(Location loc)
      : OperatorExpression(loc,
                           OperatorExpressionKind::ErrorPropagationExpression) {
  }

  std::shared_ptr<Expression> getLHS() const { return expr; };

  void setLhs(std::shared_ptr<Expression> l) { expr = l; }
};

} // namespace rust_compiler::ast
