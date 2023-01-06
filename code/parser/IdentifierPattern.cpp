#include "IdentifierPattern.h"

#include <optional>

namespace rust_compiler::parser {

std::optional<patterns::IdentifierPattern>
tryParseIdentifierPattern(std::span<lexer::Token>) {
  // FIXME
  return std::nullopt;
}

} // namespace rust_compiler::parser
