#pragma once

#include "AST/BlockExpression.h"
#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class AsyncBlockExpression : public ExpressionWithoutBlock {
  bool move;
  std::shared_ptr<BlockExpression> block;

public:
  AsyncBlockExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::AsyncBlockExpression) {}
};

} // namespace rust_compiler::ast
