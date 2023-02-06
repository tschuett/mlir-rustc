#pragma once

#include "AST/Expression.h"
#include "AST/RangeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class RangeToInclusiveExpr : public RangeExpression {
  std::shared_ptr<Expression> to;

public:
  RangeToInclusiveExpr(Location loc)
      : RangeExpression(loc, RangeExpressionKind::RangeToInclusiveExpr) {}
};

} // namespace rust_compiler::ast
