#include "BlockExpression.h"

namespace rust_compiler::lexer {

  std::optional<std::shared_ptr<ast::ExpressionWithBlock>>
tryParseBlockExpression(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::lexer
