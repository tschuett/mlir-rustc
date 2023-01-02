#include "AST/ClippyAttribute.h"

namespace rust_compiler::ast {

size_t ClippyAttribute::getTokens() {
  return 7 + 2 * lints.size() - 1; // FIXME #![warn( )] ; :: in lint
  return 7 + 2 * lintTokens - 1;
}

} // namespace rust_compiler::ast
