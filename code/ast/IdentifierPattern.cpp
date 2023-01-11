#include "AST/Patterns/IdentifierPattern.h"

namespace rust_compiler::ast::patterns {

size_t IdentifierPattern::getTokens() {
  size_t count = 0;

  if (ref)
    ++count;

  if (mut)
    ++count;

  return 1 + count;
}

std::string IdentifierPattern::getIdentifier() { return identifier; }

} // namespace rust_compiler::ast::patterns
