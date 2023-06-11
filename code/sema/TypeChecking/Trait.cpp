#include "ADT/CanonicalPath.h"
#include "AST/AssociatedItem.h"
#include "AST/Function.h"
#include "AST/GenericParam.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/TypePath.h"
#include "AST/VisItem.h"
#include "Basic/Ids.h"
#include "Coercion.h"
#include "TyCtx/TraitQueryGuard.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"
#include "TypeChecking.h"
#include "llvm/Support/ErrorHandling.h"

#include <memory>
#include <optional>
#include <vector>

using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *TypeResolver::checkTrait(ast::Trait *s) {

  TraitReference *ref = resolveTrait(s);

  if (ref->isError())
    return new TyTy::ErrorType(s->getNodeId());

  TypeIdentity ident = {CanonicalPath::createEmpty(), s->getLocation()};

  return new TyTy::DynamicObjectType(
      s->getNodeId(), ident,
      {TyTy::TypeBoundPredicate(*ref, s->getLocation())});
}

TraitReference *TypeResolver::resolveTrait(ast::Trait *trait) {
  // TraitReference *tref = &TraitReference::error_node();
  std::optional<TraitReference *> ref =
      tcx->lookupTraitReference(trait->getNodeId());
  if (ref)
    return *ref;

  if (tcx->isTraitQueryInProgress(trait->getNodeId())) {
    // report error: cycle
    assert(false);
  }

  TraitQueryGuard guard = {trait->getNodeId(), tcx};
  TyTy::BaseType *self = nullptr;
  std::vector<TyTy::SubstitutionParamMapping> substitutions;

  if (trait->hasGenericParams()) {
    for (ast::GenericParam &gp : trait->getGenericParams().getGenericParams()) {
      switch (gp.getKind()) {
      case GenericParamKind::LifetimeParam: {
        break;
      }
      case GenericParamKind::ConstParam: {
        break;
      }
      case GenericParamKind::TypeParam: {
        TypeParam tp = gp.getTypeParam();
        TyTy::ParamType *type = checkGenericParam(gp);
        tcx->insertType(gp.getIdentity(), type);
        substitutions.push_back(TyTy::SubstitutionParamMapping(tp, type));

        // if (tp.getType()
        break;
      }
      }
    }
  }

  std::vector<TyTy::TypeBoundPredicate> specifiedBounds;
  std::vector<TyTy::SubstitutionParamMapping> copy;
  for (auto &sub : substitutions)
    copy.push_back(sub.clone());

  /// FIXME

  std::vector<const TraitReference *> superTraits;
  if (trait->hasTypeParamBounds()) {
    types::TypeParamBounds bounds = trait->getTypeParamBounds();
    for (auto bound : bounds.getBounds()) {
      if (bound->getKind() == TypeParamBoundKind::TraitBound) {
        std::shared_ptr<TraitBound> tb =
            std::static_pointer_cast<TraitBound>(bound);
        TyTy::TypeBoundPredicate pred = getPredicateFromBound(tb->getPath(), nullptr);
        if (pred.isError()) {
          // report error
        }
        specifiedBounds.push_back(pred);
        superTraits.push_back(pred.get());
      }
    }
  }
  self->inheritBounds(specifiedBounds);

  std::vector<TraitItemReference> itemRefs;
  for (auto &item : trait->getAssociatedItems()) {
    std::vector<TyTy::SubstitutionParamMapping> itemSubst;
    for (auto &sub : substitutions)
      itemSubst.push_back(sub.clone());

    TraitItemReference traitItemRef =
        resolveAssociatedItemInTraitToRef(item, self, itemSubst);
    itemRefs.push_back(traitItemRef);
  }

  TraitReference traitObject = {trait, itemRefs, superTraits, substitutions};
  tcx->insertTraitReference(trait->getNodeId(), std::move(traitObject));
  std::optional<TraitReference *> ref2 =
      tcx->lookupTraitReference(trait->getNodeId());
  assert(ref2.has_value());

  size_t index = 0;
  for (const AssociatedItem &asso : trait->getAssociatedItems()) {
    if (asso.hasConstantItem()) {
      assert(false);
    } else if (asso.hasFunction()) {
      resolveFunctionItemInTrait(asso.getFunction(), itemRefs[index].getType());
    } else if (asso.hasMacroInvocationSemi()) {
      assert(false);
    } else if (asso.hasTypeAlias()) {
      assert(false);
    }
    ++index;
  }

  //(*ref2)->onResolved();

  return *ref2;
}

TraitItemReference TypeResolver::resolveAssociatedItemInTraitToRef(
    AssociatedItem &item, TyTy::BaseType *self,
    const std::vector<TyTy::SubstitutionParamMapping> &substitutions) {
  if (item.hasFunction()) {
    std::shared_ptr<Function> fun =
        std::static_pointer_cast<Function>(item.getFunction());
    return TraitItemReference(fun->getName(), fun->hasBody(),
                              TraitItemKind::Function, &item, self, // fishy
                              substitutions, item.getLocation());
  } else if (item.hasTypeAlias()) {
    assert(false);
  } else if (item.hasConstantItem()) {
    assert(false);
  }
  assert(false);
}

TyTy::ParamType *TypeResolver::checkGenericParam(const ast::GenericParam &gp) {
  switch (gp.getKind()) {
  case GenericParamKind::LifetimeParam: {
    assert(false);
    break;
  }
  case GenericParamKind::TypeParam: {
    return checkGenericParamTypeParam(gp.getTypeParam());
  }
  case GenericParamKind::ConstParam: {
    assert(false);
    break;
  }
  }
  llvm_unreachable("unkown generic param kind");
}

TyTy::ParamType *
TypeResolver::checkGenericParamTypeParam(const ast::TypeParam &tp) {
  if (tp.hasType())
    checkType(tp.getType());

  TypeExpression *implicitSelfBound= nullptr;
  if (tp.hasTypeParamBounds()) {
    basic::NodeId implicitId = getNextNodeId();
    TyTy::ParamType *p = new TyTy::ParamType(
        tp.getIdentifier(), tp.getLocation(), implicitId, tp, {});
    tcx->insertImplicitType(implicitId, p);

    implicitSelfBound = new TypePath(Location::getEmptyLocation());
  }

  std::vector<TyTy::TypeBoundPredicate> specifiedBounds;
  if (tp.hasTypeParamBounds()) {
    for (auto &bound : tp.getBounds().getBounds()) {
      switch (bound->getKind()) {
      case TypeParamBoundKind::Lifetime:
        break;
      case TypeParamBoundKind::TraitBound: {
        std::shared_ptr<TraitBound> tb =
            std::static_pointer_cast<TraitBound>(bound);
        TyTy::TypeBoundPredicate pred =
            getPredicateFromBound(tb->getPath(), implicitSelfBound);
        if (!pred.isError())
          specifiedBounds.push_back(pred);
        break;
      }
      }
    }
  }

  return new TyTy::ParamType(tp.getIdentifier(), tp.getLocation(),
                             tp.getNodeId(), tp, specifiedBounds);
}

void TypeResolver::resolveFunctionItemInTrait(std::shared_ptr<Item> item,
                                              TyTy::BaseType *type) {
  assert(item->getItemKind() == ItemKind::VisItem);
  Function *fun = std::static_pointer_cast<Function>(
                      std::static_pointer_cast<VisItem>(item))
                      .get();
  if (!fun->hasBody())
    return;
  if (type->getKind() == TyTy::TypeKind::Error)
    return;
  assert(type->getKind() == TyTy::TypeKind::Function);
  FunctionType *funType = static_cast<FunctionType *>(type);
  BaseType *returnType = funType->getReturnType();
  pushReturnType(TypeCheckContextItem(fun), returnType);

  BaseType *bodyType = checkExpression(fun->getBody());

  Location returnLoc = fun->hasReturnType()
                           ? fun->getReturnType()->getLocation()
                           : fun->getLocation();

  coercionWithSite(fun->getNodeId(), TyTy::WithLocation(returnType, returnLoc),
                   TyTy::WithLocation(bodyType, fun->getLocation()),
                   fun->getLocation(), tcx);

  popReturnType();
}

} // namespace rust_compiler::sema::type_checking
