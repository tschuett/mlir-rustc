#include "TypeBoundsProbe.h"

#include "AST/Implementation.h"
#include "AST/TraitImpl.h"
#include "AST/Types/TypeExpression.h"
#include "TraitResolver.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TraitReference.h"


#include "TypeChecking.h"

using namespace rust_compiler::adt;
using namespace rust_compiler::ast;

namespace rust_compiler::sema::TypeChecking {

std::vector<std::pair<TyTy::TraitReference *, ast::TraitImpl *>>
TypeBoundsProbe::probe(TyTy::BaseType *receiver,
                       type_checking::TypeResolver *resolver) {

  TypeBoundsProbe probe = {receiver, resolver};

  probe.scan();
  return probe.traitReferences;
}

void TypeBoundsProbe::scan() {
  std::vector<std::pair<ast::types::TypeExpression *, ast::TraitImpl *>>
      possibleTraitPaths;

  context->iterateImplementations(
      [&](NodeId id, ast::Implementation *impl) mutable -> bool {
        if (impl->getKind() == ast::ImplementationKind::InherentImpl)
          return true;
        ast::TraitImpl *trait = static_cast<ast::TraitImpl *>(impl);
        NodeId implTypeId = trait->getType()->getNodeId();
        std::optional<TyTy::BaseType *> implType =
            resolver->queryType(implTypeId);
        if (!implType)
          return true;

        if (!receiver->canEqual(*implType, false))
          if (!((*implType)->canEqual(receiver, false)))
            return true;

        possibleTraitPaths.push_back({trait->getTypePath().get(), trait});
        return true;
      });

  for (auto &path : possibleTraitPaths) {
    ast::types::TypeExpression *traitPath = path.first;
    TyTy::TraitReference *traitRef = TraitResolver::resolve(traitPath);

    if (!traitRef->isError())
      traitReferences.push_back({traitRef, path.second});
  }
}

} // namespace rust_compiler::sema::TypeChecking
