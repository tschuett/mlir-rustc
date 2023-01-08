#include "ArithmeticOrLogicalExpression.h"

#include "AST/ArithmeticOrLogicalExpression.h"
#include "AST/Expression.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

static const std::pair<TokenKind, ArithmeticOrLogicalExpressionKind>
    operators[] = {
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

  llvm::errs() << "tryParseArithmeticOrLogicalExpresion"
               << "\n";

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
        llvm::errs() << "tryParseArithmeticOrLogicalExpresion: success"
                     << "\n";

        return std::static_pointer_cast<ast::Expression>(
            std::make_shared<ArithmeticOrLogicalExpression>(
                tokens.front().getLocation(), *kind, *left, *right));
      }
    }
  }
  // FIXME

  llvm::errs() << "tryParseArithmeticOrLogicalExpresion: failedq"
               << "\n";

  return std::nullopt;
}

} // namespace rust_compiler::parser
