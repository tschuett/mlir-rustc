#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class AwaitExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> lhs;

public:
  AwaitExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::AwaitExpression) {}

  void setLhs(std::shared_ptr<Expression> lh) { lhs = lh; }
  std::shared_ptr<Expression> getBody() const;
};

} // namespace rust_compiler::ast
