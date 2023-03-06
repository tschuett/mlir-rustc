#include "AST/Module.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

  void Sema::walkOuterAttributes(std::span<ast::OuterAttribute> attr) {}

} // namespace rust_compiler::sema
