#include "Casting.h"

#include "Coercion.h"
#include "Session/Session.h"

using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *Casting::castWithSite(basic::NodeId id, TyTy::WithLocation lhs,
                                      TyTy::WithLocation rhs, Location loc) {
  Casting casting;

  return casting.cast(id, lhs, rhs, loc);
}

TyTy::BaseType *Casting::cast(basic::NodeId id, TyTy::WithLocation lhs,
                              TyTy::WithLocation rhs, Location loc) {

  TyCtx *context = rust_compiler::session::session->getTypeContext();

  Coercion coerce = {context};

  CoercionResult result = coerce.coercion(lhs.getType(), rhs.getType(), loc,
                                          true /* allowAutoderef*/);

  assert(false);
}

} // namespace rust_compiler::sema::type_checking
