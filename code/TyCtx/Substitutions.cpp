#include "TyCtx/Substitutions.h"

#include "TyCtx/TyTy.h"

namespace rust_compiler::tyctx::TyTy {

SubstitutionArgumentMappings getUsedSubstitutionArguments(TyTy::BaseType *);

std::string SubstitutionParamMapping::toString() const {
  if (param == nullptr)
    return "nullptr";

  return param->toString();
}

} // namespace rust_compiler::tyctx::TyTy
