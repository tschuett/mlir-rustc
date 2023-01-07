#include "ArithmeticOrLogicalExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseArithmeticOrLogicalExpresion(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> left =
      tryParseExpression(view);

  if (left) {
    std::optional<std::shared_ptr<ast::Expression>> right =
        tryParseExpression(view);

    if (right) {
    }
  }
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
