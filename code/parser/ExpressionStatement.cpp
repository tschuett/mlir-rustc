#include "ExpressionStatement.h"

#include "ExpressionWithBlock.h"
#include "ExpressionWithoutBlock.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseExpressionStatement(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> woBlock =
      tryParseExpressionWithoutBlock(view);

  if (woBlock) {
    view = view.subspan((*woBlock)->getTokens());

    if (view.front().getKind() == lexer::TokenKind::Semi) {
      (*woBlock)->setHasTrailingSemi();
      return *woBlock;
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

    return *withBlock;
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
