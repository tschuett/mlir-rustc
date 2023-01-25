#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class AwaitExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> body;

public:
  AwaitExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::AwaitExpression) {}

  std::shared_ptr<Expression> getBody() const;
};

} // namespace rust_compiler::ast
