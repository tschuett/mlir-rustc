#pragma once

#include "AST/OperatorExpression.h"

namespace rust_compiler::ast {

class ErrorPropagationExpression : public OperatorExpression {
public:
  ErrorPropagationExpression(Location loc)
      : OperatorExpression(loc,
                           OperatorExpressionKind::ErrorPropagationExpression) {
  }
};

} // namespace rust_compiler::ast
