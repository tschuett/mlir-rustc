#pragma once

#include "AST/OperatorExpression.h"

namespace rust_compiler::ast {

class AssignmentExpression : public OperatorExpression {
public:
  AssignmentExpression(Location loc)
      : OperatorExpression(loc, OperatorExpressionKind::AssignmentExpression){};
};

} // namespace rust_compiler::ast
