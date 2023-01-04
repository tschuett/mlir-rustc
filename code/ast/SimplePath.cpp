#include "AST/SimplePath.h"

namespace rust_compiler::ast {

void SimplePath::setWithDoubleColon() { withDoubleColon = true; }

void SimplePath::addPathSegment(SimplePathSegment &seg) {
  segments.push_back(seg);
}

size_t SimplePath::getTokens() {
  size_t tokens = 0;
  if (withDoubleColon)
    tokens += 1;

  tokens += 2 * segments.size() - 1;

  return tokens;
}

} // namespace rust_compiler::ast
