#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

std::optional<TyTy::BaseType *>
TypeResolver::resolveOperatorOverloadIndexTrait(ast::IndexExpression *index,
                                                TyTy::BaseType *arrayExprType,
                                                TyTy::BaseType *indexExprType) {
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
