#pragma once

#include "AST/Expression.h"

#include <memory>

namespace rust_compiler::ast {

class IndexExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> array;
  std::shared_ptr<Expression> index;

public:
  IndexExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::IndexExpression) {}
};

} // namespace rust_compiler::ast
