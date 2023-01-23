#include "AST/ExpressionStatement.h"
#include "AST/Statement.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Statement>>
Parser::tryParseExpressionStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);

  if (woBlock) {
    view = view.subspan((*woBlock)->getTokens());

    if (view.front().getKind() == lexer::TokenKind::Semi) {
      (*woBlock)->setHasTrailingSemi();
      return std::make_shared<ExpressionStatement>(tokens.front().getLocation(), *woBlock);
    } else {
      return std::nullopt;
    }
  }

  // then ;

  std::optional<std::shared_ptr<ast::Expression>> withBlock =
      tryParseExpressionWithBlock(view);

  if (withBlock) {
    view = view.subspan((*woBlock)->getTokens());
    if (view.front().getKind() == lexer::TokenKind::Semi) {
      (*withBlock)->setHasTrailingSemi();
    }

    return std::make_shared<ExpressionStatement>(tokens.front().getLocation(), *withBlock);
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
