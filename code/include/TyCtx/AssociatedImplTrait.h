#pragma once

#include "AST/TraitImpl.h"
#include "TyCtx/Predicate.h"
#include "TyCtx/TraitReference.h"

namespace rust_compiler::tyctx::TyTy {
class BaseType;
}

namespace rust_compiler::tyctx {

class TyCtx;

class AssociatedImplTrait {
public:
  AssociatedImplTrait(TyTy::TraitReference *trait,
                      TyTy::TypeBoundPredicate predicate, ast::TraitImpl *impl,
                      TyTy::BaseType *self,
                      rust_compiler::tyctx::TyCtx *context)
      : trait(trait), predicate(predicate), impl(impl), self(self),
        context(context){};

  ast::TraitImpl *getImplementation() const { return impl; }

  TyTy::BaseType *
  setupAssociatedTypes(TyTy::BaseType *,
                       const TyTy::TypeBoundPredicate &predicate);

  TyTy::BaseType *getSelf() const { return self; }

  ast::TraitImpl *getTraitImplementation() const { return impl; }

private:
  TyTy::TraitReference *trait;
  TyTy::TypeBoundPredicate predicate;
  ast::TraitImpl *impl;
  TyTy::BaseType *self;
  tyctx::TyCtx *context;
};

} // namespace rust_compiler::tyctx
