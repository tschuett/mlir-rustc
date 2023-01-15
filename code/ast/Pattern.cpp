#include "AST/Patterns/Pattern.h"

namespace rust_compiler::ast::patterns {

void Pattern::addPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat) {
  patterns.push_back(pat);
}

size_t Pattern::getTokens() {}

} // namespace rust_compiler::ast::patterns
