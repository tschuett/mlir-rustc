#pragma once

#include "AST/Expression.h"

#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class FieldExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expr;
  std::string identifier;

public:
  FieldExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::FieldExpression) {}

  void setLeft(std::shared_ptr<Expression> l) { expr = l; }
  void setIdentifier(std::string_view s) { identifier = s; }
};

} // namespace rust_compiler::ast
