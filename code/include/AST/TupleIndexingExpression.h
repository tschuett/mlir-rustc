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

  void setTuple(std::shared_ptr<Expression> t) { tuple = t; }
  void setIndex(std::string_view i) { index = i; }

  std::shared_ptr<Expression> getTuple() const { return tuple; }
  size_t getIndex() const {
    return atoi(index.c_str());
  }
};

} // namespace rust_compiler::ast
