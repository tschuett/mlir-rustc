#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkClosureExpression(std::shared_ptr<ast::ClosureExpression>) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
