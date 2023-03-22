#include "Coercion.h"

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *coercion(basic::NodeId, TyTy::WithLocation lhs,
                        TyTy::WithLocation rhs, Location unify) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
