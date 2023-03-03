#pragma once

#include "AST/AST.h"
#include "AST/Expression.h"
#include "Lexer/Token.h"

#include <memory>
#include <optional>

namespace rust_compiler::ast {

class BreakExpression : public ExpressionWithoutBlock {
  std::shared_ptr<Expression> expr;
  std::optional<lexer::Token> token;

public:
  BreakExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::BreakExpression){};

  void setExpression(std::shared_ptr<Expression>);
  void setLifetime(lexer::Token _token) { token = _token; }
};

} // namespace rust_compiler::ast

// FIXME LIFETIME_OR_LABEL
