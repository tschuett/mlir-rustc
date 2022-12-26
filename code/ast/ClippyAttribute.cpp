#include "AST/ClippyAttribute.h"

namespace rust_compiler::ast {

ClippyAttribute::ClippyAttribute(std::span<std::string> _lints) {
  lints = {_lints.begin(), _lints.end()};
}

size_t ClippyAttribute::getTokens() {
  return 7 + 2 * lints.size() - 1; // FIXME #![warn( )]
}

} // namespace rust_compiler::ast
