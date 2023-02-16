#include "AST/Types/RawPointerType.h"

namespace rust_compiler::ast::types {

void RawPointerType::setMut() { mut = true; }
void RawPointerType::setConst() { con = true; }

void RawPointerType::setType(
    std::shared_ptr<ast::types::TypeNoBounds> _bounds) {
  type = _bounds;
}

} // namespace rust_compiler::ast::types
