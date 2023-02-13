#include "AST/ShorthandSelf.h"

namespace rust_compiler::ast {

void ShorthandSelf::setMut() { mut = true; }
void ShorthandSelf::setAnd() { andP = true; }


} // namespace rust_compiler::ast
