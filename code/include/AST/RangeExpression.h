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
  std::optional<std::shared_ptr<ast::Expression>> left;

public:
  RangeExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::RangeExpression) {}

  RangeExpressionKind getKind() const { return kind; };
  void setKind(RangeExpressionKind k) { kind = k; }
  void setLeft(std::shared_ptr<ast::Expression> l) { left = l; }
};

} // namespace rust_compiler::ast
