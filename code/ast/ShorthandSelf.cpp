#include "AST/ShorthandSelf.h"

namespace rust_compiler::ast {

void ShorthandSelf::setMut() { mut = true; }
void ShorthandSelf::setAnd() { andP = true; }

size_t ShorthandSelf::getTokens() {
  size_t count = 0;

  if (mut)
    ++count;

  if (andP)
    ++count;

  return 1 + count;
}

} // namespace rust_compiler::ast
