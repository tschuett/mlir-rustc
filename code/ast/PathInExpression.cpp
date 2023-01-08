#include "AST/PathInExpression.h"

#include "AST/PathExprSegment.h"

#include <cassert>

namespace rust_compiler::ast {

size_t PathInExpression::getTokens() {
  size_t count = 0;

  for (PathExprSegment &seg : segs) {
    count += seg.getTokens();
  }
  count += doubleColons;

  return count;
}

} // namespace rust_compiler::ast
