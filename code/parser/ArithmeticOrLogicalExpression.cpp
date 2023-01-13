#include "ArithmeticOrLogicalExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/Expression.h"
#include "OperatorFeeding.h"
#include "Parser/Parser.h"

#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static const std::pair<TokenKind, ArithmeticOrLogicalExpressionKind> ops[] = {
    {TokenKind::Plus, ArithmeticOrLogicalExpressionKind::Addition},
    {TokenKind::Minus, ArithmeticOrLogicalExpressionKind::Subtraction},
    {TokenKind::Star, ArithmeticOrLogicalExpressionKind::Multiplication},
    {TokenKind::Slash, ArithmeticOrLogicalExpressionKind::Division},
    {TokenKind::Percent, ArithmeticOrLogicalExpressionKind::Remainder},
    {TokenKind::And, ArithmeticOrLogicalExpressionKind::BitwiseAnd},
    {TokenKind::Or, ArithmeticOrLogicalExpressionKind::BitwiseOr},
    {TokenKind::Caret, ArithmeticOrLogicalExpressionKind::BitwiseXor},
    {TokenKind::Shl, ArithmeticOrLogicalExpressionKind::LeftShift},
    {TokenKind::Shr, ArithmeticOrLogicalExpressionKind::RightShift}};

std::optional<ArithmeticOrLogicalExpressionKind>
Parser::tryParserOperator(std::span<lexer::Token> tokens) {
  if (tokens.empty())
    return std::nullopt;

  TokenKind front = tokens.front().getKind();

  //  std::string Token2String(TokenKind kind);

  for (auto &tok : ops) {
    if (front == std::get<0>(tok))
      return std::get<1>(tok);
  }

  return std::nullopt;
}

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseArithmeticOrLogicalExpresion(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  llvm::outs() << "tryParseArithmeticOrLogicalExpresion"
               << "\n";

  std::optional<std::shared_ptr<ast::Expression>> left =
      tryParseOperatorFeedingExpression(view);

  if (left) {
    view = view.subspan((*left)->getTokens());

    // find Operator
    std::optional<ArithmeticOrLogicalExpressionKind> kind =
        tryParserOperator(view);

    if (kind) {
      view = view.subspan(1);
      std::optional<std::shared_ptr<ast::Expression>> right =
          tryParseOperatorFeedingExpression(view);

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

// FIXME limit scope of left right expressions
