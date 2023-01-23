#include "Statement.h"

#include "AST/Expression.h"
#include "AST/ExpressionStatement.h"
#include "AST/LetStatement.h"
#include "AST/Statement.h"
#include "ExpressionStatement.h"
#include "Parser/Parser.h"

#include <memory>
#include <optional>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statement>>
Parser::tryParseStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Statement>> letExpr =
      tryParseLetStatement(view);

  if (letExpr) {
    view = view.subspan((*letExpr)->getTokens());

    // if (view.front().getKind() == TokenKind::Semi) {
    return *letExpr;
    //}
  }

  std::optional<std::shared_ptr<ast::Statement>> expr =
      tryParseExpressionStatement(view);

  if (expr) {
    view = view.subspan((*expr)->getTokens());

    // if (view.front().getKind() == TokenKind::Semi) {
    // ExpressionStatement stmt = {tokens.front().getLocation(), *expr};
    return *expr;
    //}
  }


  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
