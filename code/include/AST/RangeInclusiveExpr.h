#pragma once

#include "AST/Expression.h"
#include "AST/RangeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class RangeInclusiveExpr : public RangeExpression {
  std::shared_ptr<Expression> from;
  std::shared_ptr<Expression> to;

public:
  RangeInclusiveExpr(Location loc)
      : RangeExpression(loc, RangeExpressionKind::RangeInclusiveExpr) {}
};

} // namespace rust_compiler::ast
