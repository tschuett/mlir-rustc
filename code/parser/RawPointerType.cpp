#include "AST/Types/RawPointerType.h"

namespace rust_compiler::ast::types {

void RawPointerType::setMut() { mut = true; }
void RawPointerType::setConst() { con = true; }


} // namespace rust_compiler::ast::types
