#include "OperatorExpression.h"

#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseOperatorExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::Expression>> arith =
      tryParseArithmeticOrLogicalExpresion(view);

  if (arith) {
    return *arith;
  }

  return std::nullopt;
}

} // namespace rust_compiler::parser
