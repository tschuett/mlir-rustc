#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class UnsafeBlockExpression : public ExpressionWithBlock {
  std::shared_ptr<Expression> expr;

public:
  UnsafeBlockExpression(Location loc)
      : ExpressionWithBlock(
            loc, ExpressionWithBlockKind::UnsafeBlockExpression) {}

  void setBlock(std::shared_ptr<Expression>);
};

} // namespace rust_compiler::ast

// FIXME kind?
