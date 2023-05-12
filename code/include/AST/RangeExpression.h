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
  RangeFullExpr,
  RangeInclusiveExpr,
  RangeToInclusiveExpr
};

class RangeExpression : public ExpressionWithoutBlock {
  RangeExpressionKind kind;
  std::optional<std::shared_ptr<ast::Expression>> left;
  std::optional<std::shared_ptr<ast::Expression>> right;

public:
  RangeExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::RangeExpression) {}

  RangeExpressionKind getKind() const { return kind; };
  void setKind(RangeExpressionKind k) { kind = k; }
  void setLeft(std::shared_ptr<ast::Expression> l) { left = l; }
  void setRight(std::shared_ptr<ast::Expression> r) { right = r; }

  std::shared_ptr<ast::Expression> getLeft() const { return *left; }
  std::shared_ptr<ast::Expression> getRight() const { return *right; }
};

} // namespace rust_compiler::ast
