#include "TyCtx/AssociatedImplTrait.h"

#include "AST/AssociatedItem.h"
#include "AST/GenericParam.h"
#include "AST/GenericParams.h"
#include "AST/TypeAlias.h"
#include "Lexer/Identifier.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/SubstitutionsMapper.h"
#include "TyCtx/TyCtx.h"
#include "TyCtx/TyTy.h"

// FIXME
#include "../sema/TypeChecking/Unification.h"

#include <memory>
#include <optional>
#include <vector>

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::tyctx {

TyTy::BaseType *AssociatedImplTrait::setupAssociatedTypes(
    TyTy::BaseType *self, const TyTy::TypeBoundPredicate &predicate) {
  TyTy::BaseType *receiver = self->clone();
  TyTy::BaseType *associatedSelf = getSelf();
  std::vector<TyTy::SubstitutionParamMapping> substitutions;

  ast::TraitImpl *impl = getImplementation();
  if (impl->hasGenericParams()) {
    ast::GenericParams generic = impl->getGenericParams();

    for (const ast::GenericParam &g : generic.getGenericParams()) {
      switch (g.getKind()) {
      case GenericParamKind::LifetimeParam: {
        break;
      }
      case GenericParamKind::TypeParam: {
        std::optional<TyTy::BaseType *> l = context->lookupType(g.getNodeId());
        if (l)
          if ((*l)->getKind() == TypeKind::Parameter)
            substitutions.push_back(TyTy::SubstitutionParamMapping(
                g.getTypeParam(), static_cast<TyTy::ParamType *>(*l)));
        break;
      }
      case GenericParamKind::ConstParam: {
        break;
      }
      }
    }
  }
  std::map<lexer::Identifier, NodeId> paramMappings;
  TyTy::ParamSubstCallback paramSubstCb = [&](const TyTy::ParamType &p,
                                              const TyTy::SubstitutionArg &a) {
    paramMappings[p.getSymbol()] = a.getType()->getReference();
  };

  Location loc;
  std::vector<TyTy::SubstitutionArg> args;
  for (const TyTy::SubstitutionParamMapping &p : substitutions) {
    if (p.needSubstitution()) {
      TyTy::TypeVariable inferVar =
          TyTy::TypeVariable::getImplicitInferVariable(loc);
      args.push_back(TyTy::SubstitutionArg(&p, inferVar.getType()));
    } else {
      TyTy::ParamType *param = p.getParamType();
      TyTy::BaseType *resolved = param->destructure();
      args.push_back(TyTy::SubstitutionArg(&p, resolved));
      paramMappings[param->getSymbol()] = resolved->getReference();
    }
  }

  TyTy::SubstitutionArgumentMappings inferArguments = {
      std::move(args), {}, loc, paramSubstCb};

  InternalSubstitutionsMapper intern;
  TyTy::BaseType *implSelfInfer =
      (!associatedSelf->isConcrete())
          ? intern.resolve(associatedSelf, inferArguments)
          : associatedSelf;

  TyTy::TypeBoundPredicate implPredicate =
      associatedSelf->getSpecifiedBounds()[0];

  std::vector<TyTy::BaseType *> implTraitPredicateArgs;
  for (size_t i = 0; i < implPredicate.getNumberOfSubstitutions(); ++i) {
    const auto &arg = implPredicate.getSubstitutions()[i];
    if (i == 0)
      continue;

    TyTy::ParamType *p = arg.getParamType();
    TyTy::BaseType *r = p->resolve();
    if (!r->isConcrete()) {
      InternalSubstitutionsMapper intern;
      r = intern.resolve(r, inferArguments);
    }

    implTraitPredicateArgs.push_back(r);
  }

  std::vector<TyTy::BaseType *> hrtbBoundArguments;
  for (size_t i = 0; i < predicate.getNumberOfSubstitutions(); ++i) {
    auto &arg = predicate.getSubstitutions()[i];
    if (i == 0)
      continue;
    TyTy::ParamType *p = arg.getParamType();
    TyTy::BaseType *r = p->resolve();
    InternalSubstitutionsMapper intern;
    if (!r->isConcrete())
      r = intern.resolve(r, inferArguments);
    hrtbBoundArguments.push_back(r);
  }

  assert(implTraitPredicateArgs.size() == hrtbBoundArguments.size());

  for (size_t i = 0; i < implTraitPredicateArgs.size(); ++i) {
    TyTy::BaseType *a = implTraitPredicateArgs[i];
    TyTy::BaseType *b = hrtbBoundArguments[i];

    TyTy::BaseType *result = Unification::unifyWithSite(
        TyTy::WithLocation(a), TyTy::WithLocation(b), loc, context);
    assert(result->getKind() != TypeKind::Error);
  }

  TyTy::BaseType *result = Unification::unifyWithSite(
      TyTy::WithLocation(receiver), TyTy::WithLocation(implSelfInfer), loc,
      context);
  assert(result->getKind() != TypeKind::Error);

  TyTy::BaseType *selfResult = result;

  std::vector<TyTy::SubstitutionArg> associatedArguments;
  for (auto &p : substitutions) {
    Identifier symbol = p.getParamType()->getSymbol();
    auto it = paramMappings.find(symbol);

    NodeId id = it->second;
    std::optional<TyTy::BaseType *> argument = context->lookupType(id);
    assert(argument.has_value());
    TyTy::SubstitutionArg arg = {&p, *argument};
    associatedArguments.push_back(arg);
  }

  TyTy::SubstitutionArgumentMappings associatedTypeArgs = {
      std::move(associatedArguments), {}, loc};

  for (auto &asso : impl->getAssociatedItems()) {
    switch (asso.getKind()) {
    case AssociatedItemKind::MacroInvocationSemi: {
      break;
    }
    case AssociatedItemKind::TypeAlias: {
      ast::TypeAlias *alias =
          std::static_pointer_cast<ast::TypeAlias>(asso.getTypeAlias()).get();
      std::optional<TraitItemReference *> resolvedTraitItem =
          trait->lookupTraitItem(alias->getIdentifier());
      if (!resolvedTraitItem)
        continue;
      std::optional<TyTy::BaseType *> lookup =
          context->lookupType(alias->getNodeId());
      if (!lookup)
        continue;

      InternalSubstitutionsMapper intern;
      TyTy::BaseType *substitued = intern.resolve(*lookup, associatedTypeArgs);
      (*resolvedTraitItem)->associatedTypeSet(substitued);
      break;
    }
    case AssociatedItemKind::ConstantItem: {
      break;
    }
    case AssociatedItemKind::Function: {
      break;
    }
    }
  }

  return selfResult;
}

} // namespace rust_compiler::tyctx
