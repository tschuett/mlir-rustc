#pragma once

#include "AST/Expression.h"
#include "AST/RangeExpression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

class RangeFullExpr : public RangeExpression {

public:
  RangeFullExpr(Location loc)
      : RangeExpression(loc, RangeExpressionKind::RangeFullExpr) {}
};

} // namespace rust_compiler::ast
