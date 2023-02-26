#pragma once

#include "AST/ArrayElements.h"
#include "AST/Expression.h"

#include <vector>
#include <optional>

namespace rust_compiler::ast {

class ArrayExpression : public ExpressionWithoutBlock {
  std::optional<ArrayElements> elements;

public:
  ArrayExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ArrayExpression) {}

  void setElements(const ArrayElements &a) { elements = a; }
};

} // namespace rust_compiler::ast
