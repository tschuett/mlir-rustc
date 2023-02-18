#pragma once

#include "AST/ClosureParameters.h"
#include "AST/Expression.h"
#include "AST/Types/TypeNoBounds.h"

#include <optional>
#include <variant>

namespace rust_compiler::ast {

enum RangeExpressionKind {
  RangeExpr,
  RangeFromExpr,
  RangeToExpr,
  RangeInclusiveExpr,
  RangeToInclusiveExpr
};

class RangeExpression : public ExpressionWithoutBlock {
  RangeExpressionKind kind;

public:
  RangeExpression(Location loc, RangeExpressionKind kind)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::RangeExpression),
        kind(kind) {}

  RangeExpressionKind getKind() const { return kind; };
};

} // namespace rust_compiler::ast
