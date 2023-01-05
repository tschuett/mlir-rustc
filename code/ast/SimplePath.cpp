#include "AST/SimplePath.h"

#include "AST/SimplePathSegment.h"

#include <sstream>

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

std::string SimplePath::toString() {
  std::stringstream s;

  for (SimplePathSegment &seg : segments) {
    s << seg.getSegment() << "::";
  }

  return s.str();
}

} // namespace rust_compiler::ast
