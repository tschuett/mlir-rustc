#include "ArithmeticOrLogicalExpression.h"

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static const TokenKind operators[] = {
    TokenKind::Plus,    TokenKind::Minus, TokenKind::Star, TokenKind::Slash,
    TokenKind::Percent, TokenKind::And,   TokenKind::Or,   TokenKind::Caret,
    TokenKind::Shl,     TokenKind::Shr};

std::optional<Token> tryParserOperator(std::span<lexer::Token> tokens) {
  TokenKind front = tokens.front().getKind();

  for (TokenKind tok : operators) {
    if (front == tok)
      return tokens.front();
  }

  return std::nullopt;
}

std::optional<std::shared_ptr<ast::Expression>>
tryParseArithmeticOrLogicalExpresion(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> left =
      tryParseExpression(view);

  if (left) {
    view = view.subspan((*left)->getTokens());

    // find Operator
    std::optional<Token> op = tryParserOperator(view);
    if (op) {
      view = view.subspan(1);
      std::optional<std::shared_ptr<ast::Expression>> right =
          tryParseExpression(view);

      if (right) {
      }
    }
  }
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
