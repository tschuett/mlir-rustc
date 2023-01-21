#include "LiteralExpression.h"

#include "AST/LiteralExpression.h"
#include "Lexer/KeyWords.h"
#include "Lexer/Token.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseLiteralExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  Location loc = view.front().getLocation();
  if (view.front().getKind() == TokenKind::DecLiteral) {
    return std::static_pointer_cast<ast::Expression>(
        std::make_shared<LiteralExpression>(
            loc, LiteralExpressionKind::IntegerLiteral,
            view.front().getIdentifier()));
  }

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_TRUE) {
    return std::static_pointer_cast<ast::Expression>(
        std::make_shared<LiteralExpression>(loc, LiteralExpressionKind::True,
                                            view.front().getIdentifier()));
  }

  if (view.front().isKeyWord() and
      view.front().getKeyWordKind() == KeyWordKind::KW_FALSE) {
    return std::static_pointer_cast<ast::Expression>(
        std::make_shared<LiteralExpression>(loc, LiteralExpressionKind::False,
                                            view.front().getIdentifier()));
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
