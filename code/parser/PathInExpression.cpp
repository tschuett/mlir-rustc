#include "PathInExpression.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParsePathInExpression(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
