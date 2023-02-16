#include "AST/Patterns/IdentifierPattern.h"

#include <string>
#include <vector>

namespace rust_compiler::ast::patterns {

std::string IdentifierPattern::getIdentifier() { return identifier; }

} // namespace rust_compiler::ast::patterns
