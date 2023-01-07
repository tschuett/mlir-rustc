#include "LiteralExpression.h"

#include "AST/LiteralExpression.h"
#include "Lexer/Token.h"

#include <memory>

using namespace rust_compiler::lexer;
using namespace rust_compiler::ast;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseLiteralExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  Location loc = view.front().getLocation();
  if (view.front().getKind() == TokenKind::DecIntegerLiteral) {
    return std::static_pointer_cast<ast::Expression>(
        std::make_shared<LiteralExpression>(
            loc, LiteralExpressionKind::IntegerLiteral,
            view.front().getIdentifier()));
  }

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
