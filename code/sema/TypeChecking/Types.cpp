#include "TypeChecking.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkType(std::shared_ptr<ast::types::TypeExpression>) {
  assert(false && "to be implemented");
}

void TypeResolver::checkWhereClause(const ast::WhereClause &) {
  assert(false && "to be implemented");
}

void TypeResolver::checkGenericParams(
    const GenericParams &, std::vector<TyTy::SubstitutionParamMapping>&) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
