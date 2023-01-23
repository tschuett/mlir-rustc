#include "AST/Patterns/IdentifierPattern.h"

#include <string>
#include <vector>

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

std::vector<std::string> IdentifierPattern::getLiterals() {
  std::vector<std::string> lit;
  lit.push_back(identifier);

  return lit;
}

} // namespace rust_compiler::ast::patterns
