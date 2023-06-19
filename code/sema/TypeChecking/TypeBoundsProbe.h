#pragma once

#include "AST/Implementation.h"
#include "AST/TraitImpl.h"
#include "Session/Session.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"

namespace rust_compiler::sema::TypeChecking {

using namespace rust_compiler::tyctx;

class TypeBoundsProbe {
public:
  static std::vector<std::pair<TyTy::TraitReference *, ast::TraitImpl *>>
  probe(TyTy::BaseType *receiver, type_checking::TypeResolver *resolver);

protected:
  TypeBoundsProbe(TyTy::BaseType *receiver,
                  type_checking::TypeResolver *resolver)
      : receiver(receiver), resolver(resolver) {
    context = rust_compiler::session::session->getTypeContext();
  };

  TyTy::BaseType *receiver;
  std::vector<std::pair<TyTy::TraitReference *, ast::TraitImpl *>>
      traitReferences;

  void scan();

private:
  type_checking::TypeResolver *resolver;
  tyctx::TyCtx *context;
};

} // namespace rust_compiler::sema::TypeChecking
