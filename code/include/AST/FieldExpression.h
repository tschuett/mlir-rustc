#pragma once

#include "AST/Expression.h"
#include "Lexer/Identifier.h"

#include <memory>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class FieldExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expr;
  Identifier identifier;

public:
  FieldExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::FieldExpression) {}

  void setLeft(std::shared_ptr<Expression> l) { expr = l; }
  void setIdentifier(const Identifier &id) { identifier = id; }

  Identifier getIdentifier() const { return identifier; }
  std::shared_ptr<Expression> getField() const { return expr; }
};

} // namespace rust_compiler::ast
