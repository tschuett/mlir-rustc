#include "AST/SimplePath.h"

#include "AST/SimplePathSegment.h"

#include <sstream>

namespace rust_compiler::ast {

void SimplePath::setWithDoubleColon() { withDoubleColon = true; }

void SimplePath::addPathSegment(SimplePathSegment &seg) {
  segments.push_back(seg);
}

std::string SimplePath::toString() {
  std::stringstream s;

  for (SimplePathSegment &seg : segments) {
    s << seg.getSegment() << "::";
  }

  return s.str();
}

} // namespace rust_compiler::ast
