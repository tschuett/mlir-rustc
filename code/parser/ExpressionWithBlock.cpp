#include "ExpressionWithBlock.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::Expression>>
tryParseExpressionWithBlock(std::span<lexer::Token> view) {

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
