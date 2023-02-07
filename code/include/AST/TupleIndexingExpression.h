#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class TupleIndexingExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> tuple;
  uint32_t tupleIndex;

public:
  TupleIndexingExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::TupleIndexingExpression){};
};

} // namespace rust_compiler::ast