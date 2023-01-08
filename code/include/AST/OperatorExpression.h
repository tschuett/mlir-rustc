#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

class OperatorExpression : public ExpressionWithoutBlock {
public:
  OperatorExpression(Location loc) : ExpressionWithoutBlock(loc) {}
};

} // namespace rust_compiler::ast
