#include "TypeChecking.h"

using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

std::set<MethodCandidate>
TypeResolver::resolveMethodProbe(TyTy::BaseType *receiver,
                                 TyTy::FunctionTrait) {
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
