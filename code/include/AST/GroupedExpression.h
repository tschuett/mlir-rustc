#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class GroupedExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expression;

public:
  GroupedExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::GroupedExpression) {}

  void setExpression(std::shared_ptr<Expression> e) { expression = e; }
  std::shared_ptr<Expression> getExpression() const { return expression; }
};

} // namespace rust_compiler::ast
