#pragma once

#include "AST/Expression.h"
#include "AST/RangeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class RangeToExpr : public RangeExpression {
  std::shared_ptr<Expression> to;

public:
  RangeToExpr(Location loc)
      : RangeExpression(loc, RangeExpressionKind::RangeToExpr) {}
};

} // namespace rust_compiler::ast
