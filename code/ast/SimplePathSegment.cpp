#include "AST/SimplePathSegment.h"

namespace rust_compiler::ast {

size_t SimplePathSegment::getTokens() {
  return 1; // a string
}

} // namespace rust_compiler::ast
