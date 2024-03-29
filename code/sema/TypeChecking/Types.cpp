#include "AST/Types/Types.h"

#include "ADT/CanonicalPath.h"
#include "AST/ConstParam.h"
#include "AST/Enumeration.h"
#include "AST/GenericArg.h"
#include "AST/GenericArgs.h"
#include "AST/TypeBoundWhereClauseItem.h"
#include "AST/Types/ArrayType.h"
#include "AST/Types/ReferenceType.h"
#include "AST/Types/SliceType.h"
#include "AST/Types/TraitBound.h"
#include "AST/Types/TypeExpression.h"
#include "AST/Types/TypeNoBounds.h"
#include "AST/Types/TypeParamBound.h"
#include "AST/Types/TypeParamBounds.h"
#include "AST/Types/TypePath.h"
#include "AST/WhereClauseItem.h"
#include "Basic/Ids.h"
#include "Coercion.h"
#include "PathProbing.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/SubstitutionsMapper.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/TypeIdentity.h"
#include "TypeChecking.h"
#include "TyCtx/Unification.h"

// FIXME
#include "../Resolver/Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <tuple>

using namespace rust_compiler::ast;
using namespace rust_compiler::ast::types;
using namespace rust_compiler::basic;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkType(std::shared_ptr<ast::types::TypeExpression> te) {
  std::optional<TyTy::BaseType *> type = tcx->lookupType(te->getNodeId());
  if (type)
    return *type;

  TyTy::BaseType *result = nullptr;
  switch (te->getKind()) {
  case TypeExpressionKind::TypeNoBounds: {
    result = checkTypeNoBounds(std::static_pointer_cast<TypeNoBounds>(te));
    break;
  }
  case TypeExpressionKind::ImplTraitType: {
    assert(false && "to be implemented");
  }
  case TypeExpressionKind::TraitObjectType: {
    assert(false && "to be implemented");
  }
  }
  assert(result);
  assert(result->getKind() != TyTy::TypeKind::Error);
  tcx->insertType(te->getIdentity(), result);
  return result;
}

void TypeResolver::checkWhereClause(const ast::WhereClause &where) {
  for (auto &wh : where.getItems()) {
    switch (wh->getKind()) {
    case WhereClauseItemKind::LifetimeWhereClauseItem: {
      assert(false);
      break;
    }
    case WhereClauseItemKind::TypeBoundWherClauseItem: {
      auto typeItem = std::static_pointer_cast<TypeBoundWhereClauseItem>(wh);
      auto type = typeItem->getType();
      TyTy::BaseType *binding = checkType(type);

      std::vector<TyTy::TypeBoundPredicate> specifiedBounds;
      for (auto &bound : typeItem->getBounds().getBounds()) {
        switch (bound->getKind()) {
        case TypeParamBoundKind::Lifetime: {
          break;
        }
        case TypeParamBoundKind::TraitBound: {
          auto traitBound = std::static_pointer_cast<TraitBound>(bound);
          TyTy::TypeBoundPredicate pred =
              getPredicateFromBound(traitBound->getPath(), type.get());
          if (!pred.isError())
            specifiedBounds.push_back(std::move(pred));
        }
        }
      }
      binding->inheritBounds(specifiedBounds);

      NodeId nodeId = type->getNodeId();
      std::optional<NodeId> refType = tcx->lookupResolvedType(nodeId);
      assert(refType.has_value());

      std::optional<TyTy::BaseType *> lookup = tcx->lookupType(*refType);
      assert(lookup.has_value());
      (*lookup)->inheritBounds(specifiedBounds);
    }
    }
  }
}

void TypeResolver::checkGenericParams(
    const GenericParams &pa,
    std::vector<TyTy::SubstitutionParamMapping> &substitutions) {
  for (GenericParam &param : pa.getGenericParams()) {
    switch (param.getKind()) {
    case GenericParamKind::LifetimeParam: {
      // We do not type check lifetimes
      break;
    }
    case GenericParamKind::TypeParam: {
      // We cannot type check TypeParam locally. The actually type
      // will only be known during instantiation, i.e, TypePath with
      // GenericArgs
      TypeParam pa = param.getTypeParam();
      TyTy::ParamType *paramType = checkTypeParam(pa);
      tcx->insertType(pa.getIdentity(), paramType);
      // for every TypeParam, we add one Maping
      substitutions.push_back(TyTy::SubstitutionParamMapping(pa, paramType));
      break;
    }
    case GenericParamKind::ConstParam: {
      // can type check ConstParam locally
      ConstParam cp = param.getConstParam();
      TyTy::BaseType *specifiedType = checkType(cp.getType());
      if (cp.hasLiteral() or cp.hasBlock()) {
        TyTy::BaseType *expressionType = nullptr;
        if (cp.hasLiteral())
          expressionType = checkExpression(cp.getLiteral());
        else if (cp.hasBlock())
          expressionType = checkExpression(cp.getBlock());
        coercionWithSite(cp.getNodeId(), TyTy::WithLocation(specifiedType),
                         TyTy::WithLocation(expressionType, cp.getLocation()),
                         cp.getLocation(), tcx);
      }
      tcx->insertType(cp.getIdentity(), specifiedType);
      break;
    }
    }
  }
}

TyTy::BaseType *
TypeResolver::checkTypeNoBounds(std::shared_ptr<ast::types::TypeNoBounds> no) {
  switch (no->getKind()) {
  case TypeNoBoundsKind::ParenthesizedType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::ImplTraitType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::ImplTraitTypeOneBound: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::TraitObjectTypeOneBound: {
    return checkTypeTraitObjectTypeOneBound(
        std::static_pointer_cast<TraitObjectTypeOneBound>(no));
  }
  case TypeNoBoundsKind::TypePath: {
    return checkTypePath(std::static_pointer_cast<TypePath>(no));
  }
  case TypeNoBoundsKind::TupleType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::NeverType: {
    return checkNeverType(std::static_pointer_cast<ast::types::NeverType>(no));
  }
  case TypeNoBoundsKind::RawPointerType: {
    return checkRawPointerType(
        std::static_pointer_cast<ast::types::RawPointerType>(no));
  }
  case TypeNoBoundsKind::ReferenceType: {
    return checkReferenceType(
        std::static_pointer_cast<ast::types::ReferenceType>(no));
  }
  case TypeNoBoundsKind::ArrayType: {
    return checkArrayType(std::static_pointer_cast<ast::types::ArrayType>(no));
  }
  case TypeNoBoundsKind::SliceType: {
    return checkSliceType(std::static_pointer_cast<ast::types::SliceType>(no));
  }
  case TypeNoBoundsKind::InferredType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::QualifiedPathInType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::BareFunctionType: {
    assert(false && "to be implemented");
  }
  case TypeNoBoundsKind::MacroInvocation: {
    assert(false && "to be implemented");
  }
  }
  assert(false && "to be implemented");
}

TyTy::BaseType *
TypeResolver::checkTypePath(std::shared_ptr<ast::types::TypePath> tp) {
  assert(!tp->hasLeadingPathSep() && "to be implemented");

  size_t offset = 0;
  NodeId resolvedNodeId = UNKNOWN_NODEID;
  TyTy::BaseType *root = resolveRootPathType(tp, &offset, &resolvedNodeId);
  if (root->getKind() == TyTy::TypeKind::Error) {
    llvm::errs() << "resolve root path type failed: "
                 << tp->getLocation().toString() << "\n";
    assert(false);
    return new TyTy::ErrorType(tp->getNodeId());
  }

  TyTy::BaseType *pathType = root->clone();

  pathType->setReference(tp->getNodeId());
  tcx->insertImplicitType(tp->getNodeId(), pathType);

  if (offset >= tp->getSegments().size())
    return root;

  TyTy::BaseType *segments =
      resolveSegmentsType(resolvedNodeId, tp->getNodeId(), tp, offset, root);
  assert(segments);
  assert(segments->getKind() != TyTy::TypeKind::Error);
  return segments;
}

TyTy::BaseType *
TypeResolver::resolveRootPathType(std::shared_ptr<ast::types::TypePath> path,
                                  size_t *offset,
                                  basic::NodeId *resolvedNodeId) {
  // assert(false && "to be implemented");

  TyTy::BaseType *rootType = nullptr;
  *offset = 0;

  std::vector<TypePathSegment> segs = path->getSegments();

  if (segs.size() == 1)
    if (auto t = tcx->lookupBuiltin(segs[0].getSegment().toString())) {
      *offset = 1;
      //      llvm::errs() << path->getNodeId() << " -> " << t->toString() <<
      //      "\n"; llvm::errs() << (void*)t << "\n";
      return t;
    }

  NodeId refNodeId = UNKNOWN_NODEID;
  for (unsigned i = 0; i < segs.size(); ++i) {
    bool haveMoreSegments = i != (segs.size() - 1);
    // NodeId astNodeId = segs[i].getNodeId();

    if (auto name = resolver->lookupResolvedName(segs[i].getNodeId())) {
      //      llvm::errs() << "resolve root path: it is a name"
      //                   << "\n";
      refNodeId = *name;
    } else if (auto type = resolver->lookupResolvedType(segs[i].getNodeId())) {
      //      llvm::errs() << "resolve root path: it is a type"
      //                   << "\n";
      refNodeId = *type;
    }

    if (refNodeId == UNKNOWN_NODEID) {
      if (*offset == 0) { // root
        // report error
        llvm::errs() << "unknown reference for resolved name: "
                     << segs[i].getSegment().toString() << " @ "
                     << segs[i].getLocation().toString() << "\n";
        return new TyTy::ErrorType(path->getNodeId());
      }
      return rootType;
    }

    bool segIsModule = nullptr != tcx->lookupModule(refNodeId);
    bool segIsCrate = tcx->isCrate(refNodeId);
    if (segIsModule || segIsCrate) {
      if (haveMoreSegments) {
        ++(*offset);
        continue;
      }

      // report error
      llvm::errs() << "expected value, but found crate or module "
                   << "\n";
      return new TyTy::ErrorType(path->getNodeId());
    }

    std::optional<TyTy::BaseType *> result = queryType(refNodeId); // astNodeId
    if (!result) {
      if (*offset == 0) { // root
        // report error
        llvm::errs() << "queryType failed: failed to resolve root segment"
                     << "\n";
        return new TyTy::ErrorType(path->getNodeId());
      }
      return rootType;
    }
    TyTy::BaseType *lookup = *result;

    if (rootType != nullptr) {
      // such as: GenericStruct::<_>::new(123, 456)
      if (lookup->needsGenericSubstitutions()) {
        if (!rootType->needsGenericSubstitutions()) {
          TyTy::SubstitutionArgumentMappings args =
              getUsedSubstitutionArguments(rootType);
          InternalSubstitutionsMapper mapper;
          lookup = mapper.resolve(lookup, args);
        }
      }
    }

    // turbo-fish segment path::<ty>
    // if (segs[i].hasGenerics()) {
    //  if (lookup->needsGenericSubstitutions()) {
    //    checkGenericParamsAndArgs(lookup, segs[i].getGenericArgs());
    //  }
    //} else if (lookup->needsGenericSubstitutions()) {
    //  // FIXME
    //}

    if (segs[i].hasGenerics()) {
      GenericArgs args = segs[i].getGenericArgs();
      SubstitutionsMapper mapper;
      lookup = mapper.resolve(lookup, path->getLocation(), this, &args);
      if (lookup->getKind() == TyTy::TypeKind::Error)
        return new TyTy::ErrorType(segs[i].getNodeId());
    } else if (lookup->needsGenericSubstitutions()) {
      GenericArgs args = GenericArgs::empty();
      SubstitutionsMapper mapper;
      lookup = mapper.resolve(lookup, path->getLocation(), this, &args);
    }

    *resolvedNodeId = refNodeId;
    *offset = *offset + 1;
    rootType = lookup;
  }

  return rootType;
}

TyTy::BaseType *
TypeResolver::resolveSegmentsType(basic::NodeId rootResolvedNodeId,
                                  basic::NodeId pathNodeId,
                                  std::shared_ptr<ast::types::TypePath> tp,
                                  size_t offset, TyTy::BaseType *pathType) {
  assert(false && "to be implemented");

  // basic::NodeId resolvedNodeId = rootResolvedNodeId;
  TyTy::BaseType *prevSegment = pathType;

  std::vector<TypePathSegment> segs = tp->getSegments();

  for (unsigned i = offset; i < segs.size(); ++i) {
    // segs[i];

    bool receiverIsGeneric =
        prevSegment->getKind() == TyTy::TypeKind::Parameter;
    bool probeBounds = true;
    bool probeImpls = !receiverIsGeneric;
    bool ignoreTraitItems = !receiverIsGeneric;

    std::set<PathProbeCandidate> candidates = PathProbeType::probeTypePath(
        prevSegment, segs[i].getSegment().getIdentifier(), probeImpls, false,
        ignoreTraitItems, this);

    if (candidates.size() == 0) {
      candidates = PathProbeType::probeTypePath(
          prevSegment, segs[i].getSegment().getIdentifier(), false, probeBounds,
          ignoreTraitItems, this);

      if (candidates.size() == 0) {
        // report error
        return new TyTy::ErrorType(tp->getNodeId());
      }
    }

    if (candidates.size() > 1) {
      // report error
      return new TyTy::ErrorType(tp->getNodeId());
    }

    // FIXME
  }
}

TyTy::BaseType *
TypeResolver::checkNeverType(std::shared_ptr<ast::types::NeverType>) {
  std::optional<TyTy::BaseType *> never = tcx->lookupBuiltin("!");
  if (never)
    return *never;
  return nullptr;
}

TyTy::TypeBoundPredicate TypeResolver::getPredicateFromBound(
    std::shared_ptr<ast::types::TypeExpression> path,
    ast::types::TypeExpression *associatedSelf) {
  std::shared_ptr<ast::types::TypePath> typePath =
      std::static_pointer_cast<TypePath>(path);
  std::optional<TyTy::TypeBoundPredicate> lookup =
      tcx->lookupPredicate(path->getNodeId());
  if (lookup)
    return *lookup;

  TyTy::TraitReference *trait = resolveTraitPath(typePath);
  if (trait->isError())
    return TyTy::TypeBoundPredicate::error();

  TyTy::TypeBoundPredicate predicate(*trait, path->getLocation());
  GenericArgs args = {typePath->getLocation()};

  TypePathSegment lastSegment =
      typePath->getSegments()[typePath->getSegments().size() - 1];

  if (lastSegment.hasTypeFunction()) {
    assert(false);
  } else if (lastSegment.hasGenerics()) {
    if (lastSegment.hasGenerics())
      args = lastSegment.getGenericArgs();
  }

  if (associatedSelf != nullptr) {
    GenericArgs tmp = args;
    args = GenericArgs(path->getLocation());
    GenericArg ga = {path->getLocation()};
    ga.setType(std::make_shared<ast::types::TypeExpression>(*associatedSelf));
    args.addArg(ga);
    for (const GenericArg &ga : tmp.getArgs())
      args.addArg(ga);
  }

  if (!args.isEmpty() || predicate.requiresGenericArgs())
    predicate.applyGenericArgs(&args, associatedSelf != nullptr, this);

  tcx->insertResolvedPredicate(typePath->getNodeId(), predicate);

  return predicate;
}

TyTy::ParamType *TypeResolver::checkTypeParam(const TypeParam &type) {
  if (type.hasType())
    checkType(type.getType());

  std::vector<TyTy::TypeBoundPredicate> specifiedBounds;
  if (type.hasTypeParamBounds()) {
    for (std::shared_ptr<TypeParamBound> bound : type.getBounds().getBounds()) {
      switch (bound->getKind()) {
      case TypeParamBoundKind::Lifetime: {
        break;
      }
      case TypeParamBoundKind::TraitBound: {
        std::shared_ptr<TraitBound> b =
            std::static_pointer_cast<TraitBound>(bound);
        TyTy::TypeBoundPredicate predicate =
            getPredicateFromBound(b->getPath(), nullptr);
        if (!predicate.isError())
          specifiedBounds.push_back(predicate);
        break;
      }
      }
    }
  }

  return new TyTy::ParamType(type.getIdentifier(), type.getLocation(),
                             type.getNodeId(), type, specifiedBounds);

  assert(false);
}

TyTy::BaseType *
TypeResolver::checkArrayType(std::shared_ptr<ast::types::ArrayType> arr) {
  TyTy::BaseType *capacityType = checkExpression(arr->getExpression());
  assert(capacityType->getKind() != TyTy::TypeKind::Error);

  std::optional<TyTy::BaseType *> usizeType = tcx->lookupBuiltin("usize");
  assert(usizeType.has_value());
  tcx->insertType(arr->getExpression()->getIdentity(), *usizeType);

  Unification::unifyWithSite(
      TyTy::WithLocation(*usizeType),
      TyTy::WithLocation(capacityType, arr->getExpression()->getLocation()),
      arr->getExpression()->getLocation(), tcx);

  TyTy::BaseType *base = checkType(arr->getType());

  return new TyTy::ArrayType(arr->getNodeId(), arr->getLocation(),
                             arr->getExpression(),
                             TyTy::TypeVariable(base->getReference()));
}

TyTy::SubstitutionArgumentMappings
TypeResolver::getUsedSubstitutionArguments(TyTy::BaseType *type) {
  if (type->getKind() == TyTy::TypeKind::Function)
    return static_cast<TyTy::FunctionType *>(type)->getSubstitutionArguments();
  if (type->getKind() == TyTy::TypeKind::ADT)
    return static_cast<TyTy::ADTType *>(type)->getSubstitutionArguments();
  if (type->getKind() == TyTy::TypeKind::Closure)
    return static_cast<TyTy::ClosureType *>(type)->getSubstitutionArguments();

  return TyTy::SubstitutionArgumentMappings::error();
}

TyTy::BaseType *TypeResolver::checkReferenceType(
    std::shared_ptr<ast::types::ReferenceType> ref) {
  TyTy::BaseType *base = checkType(ref->getReferencedType());
  return new TyTy::RawPointerType(ref->getNodeId(),
                                  TyTy::TypeVariable(base->getReference()),
                                  ref->getMut());
}

TyTy::BaseType *TypeResolver::checkTypeTraitObjectTypeOneBound(
    std::shared_ptr<ast::types::TraitObjectTypeOneBound> trait) {
  std::vector<TyTy::TypeBoundPredicate> specifiedBounds;

  std::shared_ptr<ast::types::TypeParamBound> b = trait->getBound();
  if (b->getKind() == TypeParamBoundKind::TraitBound) {
    TyTy::TypeBoundPredicate pred = getPredicateFromBound(
        std::static_pointer_cast<TraitBound>(b)->getPath(), nullptr);
    specifiedBounds.push_back(pred);
  }

  TypeIdentity ident = {adt::CanonicalPath::createEmpty(),
                        trait->getLocation()};
  return new TyTy::DynamicObjectType(trait->getNodeId(), ident,
                                     specifiedBounds);
}

TyTy::BaseType *TypeResolver::checkRawPointerType(
    std::shared_ptr<ast::types::RawPointerType> raw) {
  TyTy::BaseType *base = checkType(raw->getType());
  return new TyTy::RawPointerType(raw->getNodeId(),
                                  TyTy::TypeVariable(base->getReference()),
                                  raw->getMutability());
}

TyTy::BaseType *
TypeResolver::checkSliceType(std::shared_ptr<ast::types::SliceType> slice) {
  TyTy::BaseType *base = checkType(slice->getType());

  return new TyTy::SliceType(slice->getNodeId(), slice->getLocation(),
                             TyTy::TypeVariable(base->getReference()));
}

} // namespace rust_compiler::sema::type_checking
