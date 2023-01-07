#include "PathExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParsePathExpression(std::span<lexer::Token> tokens) {
  //  std::span<lexer::Token> view = tokens;
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
