#pragma once

#include "AST/Expression.h"
#include "Location.h"

namespace rust_compiler::ast {

class IfLetExpression : public ExpressionWithBlock {

public:
  IfLetExpression(Location loc)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::IfLetExpression) {}
};

} // namespace rust_compiler::ast
