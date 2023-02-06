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

};

} // namespace rust_compiler::ast

// FIXME LIFETIME_OR_LABEL
