#include "BlockExpression.h"

namespace rust_compiler::lexer {

std::optional<ast::BlockExpression>
tryParseBlockExpression(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::lexer
