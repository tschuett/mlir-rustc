#include "Statement.h"

#include "AST/Expression.h"
#include "AST/Statement.h"
#include "ExpressionStatement.h"
#include "AST/ExpressionStatement.h"

#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statement>>
tryParseStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> expr =
      tryParseExpressionStatement(view);

  if (expr) {
    view = view.subspan((*expr)->getTokens());

    if (view.front().getKind() == TokenKind::SemiColon) {
      return std::make_shared<ExpressionStatement>(tokens.front().getLocation(), *expr);
    }
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
