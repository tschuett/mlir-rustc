#include "NegationExpression.h"

#include "AST/Expression.h"
#include "AST/NegationExpression.h"

#include <memory>

using namespace rust_compiler::lexer;

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseNegationExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;
  Location loc = tokens.front().getLocation();

  if (view.front().getKind() == TokenKind::Minus) {

    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);
    if (expr) {
      NegationExpression neg = {loc};
      neg.setMinus();
      neg.setRight(*expr);

      return std::static_pointer_cast<ast::Expression>(
          std::make_shared<NegationExpression>(neg));
    }
  }

  if (view.front().getKind() == TokenKind::Not) {

    std::optional<std::shared_ptr<ast::Expression>> expr =
        tryParseExpression(view);

    if (expr) {
      NegationExpression neg = {loc};
      neg.setNot();
      neg.setRight(*expr);

      return std::static_pointer_cast<ast::Expression>(
          std::make_shared<NegationExpression>(neg));
    }
  }

  // FIXME

  return std::nullopt;
}

} // namespace rust_compiler::parser
