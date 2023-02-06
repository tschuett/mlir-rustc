
#pragma once

#include "AST/Expression.h"
#include "AST/RangeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class RangeExpr : public RangeExpression {
  std::shared_ptr<Expression> from;
  std::shared_ptr<Expression> to;

public:
  RangeExpr(Location loc)
      : RangeExpression(loc, RangeExpressionKind::RangeExpr) {}
};

} // namespace rust_compiler::ast
