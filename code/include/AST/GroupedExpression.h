#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

class GroupedExpression : public ExpressionWithoutBlock {
public:
  GroupedExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::GroupedExpression) {}
};

} // namespace rust_compiler::ast
