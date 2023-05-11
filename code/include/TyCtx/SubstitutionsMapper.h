#pragma once

#include "AST/GenericArgs.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TyTy.h"

namespace rust_compiler::tyctx {

class InternalSubstitutionsMapper {
public:
  TyTy::BaseType *resolve(TyTy::BaseType *base,
                          TyTy::SubstitutionArgumentMappings &mappings);

private:
};

class SubstitutionsMapper {
public:
  TyTy::BaseType *resolve(TyTy::BaseType *base, Location loc,
                          ast::GenericArgs *generics = nullptr);

private:
};
} // namespace rust_compiler::tyctx
