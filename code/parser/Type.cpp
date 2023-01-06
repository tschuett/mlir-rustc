#include "Type.h"

namespace rust_compiler::parser {

std::optional<std::shared_ptr<ast::types::Type>>
tryParseType(std::span<lexer::Token> tokens) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
