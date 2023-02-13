#include "AST/Patterns/IdentifierPattern.h"

#include <string>
#include <vector>

namespace rust_compiler::ast::patterns {

std::string IdentifierPattern::getIdentifier() { return identifier; }

std::vector<std::string> IdentifierPattern::getLiterals() {
  std::vector<std::string> lit;
  lit.push_back(identifier);

  return lit;
}

} // namespace rust_compiler::ast::patterns
