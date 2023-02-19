#pragma once

#include "AST/Expression.h"
#include "AST/LifetimeOrLabel.h"

#include "Lexer/Token.h"

#include <optional>

namespace rust_compiler::ast {

class ContinueExpression : public ExpressionWithoutBlock {
  //std::optional<LifetimeOrLabel> lifetimeOrLabel;
  std::optional<lexer::Token> lifetimeOrLabel;

public:
  ContinueExpression(Location loc)
      : ExpressionWithoutBlock(
            loc, ExpressionWithoutBlockKind::ContinueExpression) {}

  void setLifetime(const lexer::Token &orl) { lifetimeOrLabel = orl;}
};

} // namespace rust_compiler::ast
