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
                          sema::type_checking::TypeResolver *resolver,
                          ast::GenericArgs *generics = nullptr);

  TyTy::BaseType *resolve(TyTy::BaseType *base, Location loc);

private:
  ast::GenericArgs *generics = nullptr;
  Location loc;
};

} // namespace rust_compiler::tyctx
