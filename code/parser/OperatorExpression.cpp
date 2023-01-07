#include "OperatorExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseOperatorExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> arith =
      tryParseArithmeticOrLogicalExpresion(view);

  return std::nullopt;
}

} // namespace rust_compiler::parser
