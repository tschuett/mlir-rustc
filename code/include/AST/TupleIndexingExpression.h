#pragma once

#include "AST/Expression.h"

#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class TupleIndexingExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> tuple;
  std::string index;

public:
  TupleIndexingExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::TupleIndexingExpression){};

  void setTuple(std::shared_ptr<Expression> t) { t = tuple; }
  void setIndex(std::string_view i) { index = i; }
};

} // namespace rust_compiler::ast
