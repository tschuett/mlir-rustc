#include "TyCtx/Substitutions.h"

#include "TyCtx/TyTy.h"

namespace rust_compiler::tyctx::TyTy {

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *) {
  assert(false);
}

std::string SubstitutionParamMapping::toString() const {
  if (param == nullptr)
    return "nullptr";

  return param->toString();
}

bool SubstitutionParamMapping::needsSubstitution() const {
  return !param->isConcrete();
}

} // namespace rust_compiler::tyctx::TyTy
