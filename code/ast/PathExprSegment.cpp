#include "AST/PathExprSegment.h"

#include <cassert>

namespace rust_compiler::ast {

size_t PathExprSegment::getTokens() {
  size_t count = 0;
  if (generics)
    count += 1 + generics->getTokens();

  return 1 + count;
}
} // namespace rust_compiler::ast
