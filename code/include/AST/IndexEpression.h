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

  void setLeft(std::shared_ptr<Expression> l) { array = l; }
  void setRight(std::shared_ptr<Expression> r) { index = r; }

  std::shared_ptr<Expression> getArray() const { return array; }
  std::shared_ptr<Expression> getIndex() const { return index; }
};

} // namespace rust_compiler::ast
