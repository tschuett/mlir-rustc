#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class BreakExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expr;

public:
  BreakExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::BreakExpression){};

  void setExpression(std::shared_ptr<Expression>);
};

} // namespace rust_compiler::ast

// FIXME LIFETIME_OR_LABEL
