#pragma once

#include "AST/BlockExpression.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class UnderScoreExpression : public ExpressionWithoutBlock {
public:
  UnderScoreExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::UnderScoreExpression) {}
};

} // namespace rust_compiler::ast
