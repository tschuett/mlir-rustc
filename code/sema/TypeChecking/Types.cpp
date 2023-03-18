#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

std::optional<TyTy::BaseType *>
TypeResolver::checkType(std::shared_ptr<ast::types::TypeExpression>) {
  assert(false && "to be implemented");
}

void TypeResolver::checkWhereClause(const ast::WhereClause &) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
