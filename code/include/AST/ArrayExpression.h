#pragma once

#include "AST/ArrayElements.h"
#include "AST/Expression.h"

#include <optional>
#include <vector>

namespace rust_compiler::ast {

class ArrayExpression : public ExpressionWithoutBlock {
  std::optional<ArrayElements> elements;

public:
  ArrayExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::ArrayExpression) {}

  void setElements(const ArrayElements &a) { elements = a; }

  bool hasArrayElements() const { return elements.has_value(); }
  ArrayElements getArrayElements() const { return *elements; }
};

} // namespace rust_compiler::ast
