#include "AST/ShorthandSelf.h"

namespace rust_compiler::ast {

void ShorthandSelf::setMut() { mut = true; }
void ShorthandSelf::setAnd() { andP = true; }

size_t ShorthandSelf::getTokens() {
  assert(false);

  return 0;
}

} // namespace rust_compiler::ast
