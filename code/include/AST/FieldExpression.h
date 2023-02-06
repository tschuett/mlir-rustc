#pragma once

#include "AST/Expression.h"

#include <memory>
#include <string>

namespace rust_compiler::ast {

class FieldExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expr;
  std::string identifier;

public:
  FieldExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::FieldExpression) {}
};

} // namespace rust_compiler::ast
