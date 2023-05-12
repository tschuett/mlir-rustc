#pragma once

#include "AST/Expression.h"
#include "AST/TupleElements.h"

#include <optional>

namespace rust_compiler::ast {

class TupleExpression : public ExpressionWithoutBlock {
  std::optional<TupleElements> elements;

public:
  TupleExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::TupleExpression){};

  void setElements(const TupleElements &el) { elements = el; }

  bool isUnit() const { return !elements.has_value(); }

  TupleElements getElements() const { return *elements;}
};

} // namespace rust_compiler::ast
