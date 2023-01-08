#include "AST/IdentifierPattern.h"

namespace rust_compiler::ast {

size_t IdentifierPattern::getTokens() {
  size_t count = 0;

  if (ref)
    ++count;

  if (mut)
    ++count;

  return 1 + count;
}

} // namespace rust_compiler::ast
