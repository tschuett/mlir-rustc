#pragma once

#include "AST/Expression.h"

#include <vector>

namespace rust_compiler::ast {

class ArrayExpression : public ExpressionWithoutBlock {
  std::optional<ArrayElements> elements;

public:
  ArrayExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ArrayExpression) {}
};

} // namespace rust_compiler::ast
