#include "ArithmeticOrLogicalExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/Expression.h"

#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static const std::pair<TokenKind, ArithmeticOrLogicalExpressionKind>
    operators[] = {
        {TokenKind::Plus, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Minus, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Star, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Slash, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Percent, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::And, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Or, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Caret, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Shl, ArithmeticOrLogicalExpressionKind::Addition},
        {TokenKind::Shr, ArithmeticOrLogicalExpressionKind::Addition}};

std::optional<ArithmeticOrLogicalExpressionKind>
tryParserOperator(std::span<lexer::Token> tokens) {
  TokenKind front = tokens.front().getKind();

  for (auto &tok : operators) {
    if (front == std::get<0>(tok))
      return std::get<1>(tok);
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
    std::optional<ArithmeticOrLogicalExpressionKind> kind =
        tryParserOperator(view);
    if (kind) {
      view = view.subspan(1);
      std::optional<std::shared_ptr<ast::Expression>> right =
          tryParseExpression(view);

      if (right) {
        return std::static_pointer_cast<ast::Expression>(
            std::make_shared<ArithmeticOrLogicalExpression>(
                tokens.front().getLocation(), *kind, *left, *right));
      }
    }
  }
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
