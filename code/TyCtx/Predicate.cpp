#include "TyCtx/Predicate.h"

#include "Basic/Ids.h"

namespace rust_compiler::tyctx::TyTy {

TypeBoundPredicate TypeBoundsMappings::lookupPredicate(basic::NodeId id) {
  for (auto &cb : specifiedBounds) {
    if (cb.getId() == id)
      return cb;
  }

  return TypeBoundPredicate::error();
}

} // namespace rust_compiler::tyctx::TyTy
