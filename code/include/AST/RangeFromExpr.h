#pragma once

#include "AST/Expression.h"
#include "AST/RangeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class RangeFromExpr : public RangeExpression {
  std::shared_ptr<Expression> from;

public:
  RangeFromExpr(Location loc)
      : RangeExpression(loc, RangeExpressionKind::RangeFromExpr) {}
};

} // namespace rust_compiler::ast
