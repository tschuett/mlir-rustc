#include "AST/Module.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

  void Sema::walkOuterAttributes(std::shared_ptr<ast::OuterAttributes> attr) {}

} // namespace rust_compiler::sema
