#include "AST/SimplePath.h"

#include "AST/SimplePathSegment.h"

#include <sstream>

namespace rust_compiler::ast {

void SimplePath::setWithDoubleColon() { withDoubleColon = true; }

void SimplePath::addPathSegment(SimplePathSegment &seg) {
  segments.push_back(seg);
}


} // namespace rust_compiler::ast
