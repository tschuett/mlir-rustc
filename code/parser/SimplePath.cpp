#include "SimplePath.h"

#include <optional>

namespace rust_compiler::parser {

std::optional<ast::SimplePath>
tryParseSimplePath(std::span<lexer::Token> tokens) {

  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
