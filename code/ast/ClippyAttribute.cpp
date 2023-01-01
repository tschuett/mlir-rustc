#include "AST/ClippyAttribute.h"

namespace rust_compiler::ast {

size_t ClippyAttribute::getTokens() {
  return 7 + 2 * lints.size() - 1; // FIXME #![warn( )]
}

} // namespace rust_compiler::ast
