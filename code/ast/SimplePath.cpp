#include "AST/SimplePath.h"

namespace rust_compiler::ast {

void SimplePath::setWithDoubleColon() { withDoubleColon = true; }

void SimplePath::addPathSegment(SimplePathSegment &seg) {
  segments.push_back(seg);
}

size_t SimplePath::getTokens() {
  assert(false);
  return 0;
}

} // namespace rust_compiler::ast
