#include "AST/ComparisonExpression.h"

#include "Parser/Parser.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static const std::pair<TokenKind, ComparisonExpressionKind> ops[] = {
    {TokenKind::EqEq, ComparisonExpressionKind::Equal},
    {TokenKind::Ne, ComparisonExpressionKind::NotEqual},
    {TokenKind::Gt, ComparisonExpressionKind::GreaterThan},
    {TokenKind::Lt, ComparisonExpressionKind::LessThan},
    {TokenKind::Ge, ComparisonExpressionKind::GreaterThanOrEqualTo},
    {TokenKind::Le, ComparisonExpressionKind::LessThanOrEqualTo}};

std::optional<ComparisonExpressionKind>
tryParseComparisonOperator(std::span<lexer::Token> tokens) {
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
Parser::tryParseComparisonExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  assert(false);

  std::optional<std::shared_ptr<ast::Expression>> left =
      tryParseExpression(view);

  if (left) {
    view = view.subspan((*left)->getTokens());

    std::optional<ComparisonExpressionKind> ops = tryParseComparisonOperator(view);
    if (ops) {
      view = view.subspan(1);

      std::optional<std::shared_ptr<ast::Expression>> right =
          tryParseExpression(view);

      if (right) {
        return std::static_pointer_cast<ast::Expression>(
            std::make_shared<ComparisonExpression>(tokens.front().getLocation(),
                                                   *ops, *left, *right));
      }
    }
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
