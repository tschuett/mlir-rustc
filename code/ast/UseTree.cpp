#include "AST/UseTree.h"

namespace rust_compiler::ast::use_tree {

void Star::append(SimplePath path) {}

void SimplePathDoubleColonStar::setPath(SimplePath path) {}

size_t DoubleColonStar::getTokens() {
  return 2; // :: *
}

size_t Star::getTokens() {
  return 1; // *
}

size_t PathList::getTokens() {

  size_t size = 0;
  for (auto &el : elements) {
    size += el->getTokens();
  }

  return 2 + 2 * size - 1; // { ... }
}

} // namespace rust_compiler::ast::use_tree
