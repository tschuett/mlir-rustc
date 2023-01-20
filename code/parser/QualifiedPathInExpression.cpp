#include "Parser/Parser.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
Parser::tryParseQualifiedPathInExpression(std::span<lexer::Token> tokens) {
  std::span<lexer::Token> view = tokens;

  std::optional<std::shared_ptr<ast::types::Type>> pathType =
      tryParseQualifiedPathType(view);
  if (pathType) {
  }

  assert(false);
  return std::nullopt;
}

} // namespace rust_compiler::parser
