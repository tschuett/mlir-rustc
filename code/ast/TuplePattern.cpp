#include "AST/Patterns/TuplePattern.h"

namespace rust_compiler::ast::patterns {

void TuplePattern::add(std::shared_ptr<ast::patterns::TuplePatternItems> its) {
  items.push_back(its);
}

std::vector<std::string> TuplePattern::getLiterals() { assert(false); }

} // namespace rust_compiler::ast::patterns
