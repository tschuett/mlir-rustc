#include "AST/AssociatedItem.h"
#include "AST/EnumItem.h"
#include "AST/Implementation.h"
#include "AST/InherentImpl.h"
#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathIdentSegment.h"
#include "AST/PathInExpression.h"
#include "AST/TraitImpl.h"
#include "AST/Types/TypeExpression.h"
#include "Basic/Ids.h"
#include "PathProbing.h"
#include "TraitResolver.h"
#include "TyCtx/Predicate.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/SubstitutionsMapper.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"
#include "TyCtx/Unification.h"
#include "TypeChecking.h"

#include "../Resolver/Resolver.h"

#include <cstddef>
#include <cstdlib>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>
#include <vector>

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkPathExpression(std::shared_ptr<ast::PathExpression> path) {
  switch (path->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression: {
    return checkPathInExpression(
        std::static_pointer_cast<PathInExpression>(path));
  }
  case PathExpressionKind::QualifiedPathInExpression: {
    assert(false && "to be implemented");
  }
  }
}

TyTy::BaseType *TypeResolver::checkPathInExpression(
    std::shared_ptr<ast::PathInExpression> path) {

  size_t offset = -1;
  NodeId resolvedNodeId = UNKNOWN_NODEID;

  if (hasSmallSelf() && path->getSegments().size() == 1) {
    if (path->getSegments()[0].getIdent().getKind() ==
        PathIdentSegmentKind::self) {
      return getSmallSelf();
    }
  }

  TyTy::BaseType *typeSegment =
      resolveRootPathExpression(path, &offset, &resolvedNodeId);
  if (typeSegment->getKind() == TyTy::TypeKind::Error) {
    llvm::errs() << "checkPathInExpression failed: "
                 << path->getSegments()[0].getIdent().toString() << "\n";
    llvm::errs() << path->getNodeId() << "\n";
    llvm::errs() << path->getSegments().size() << "\n";
    assert(false);
    return typeSegment;
  }

  if (offset == path->getSegments().size())
    return typeSegment;

  std::vector<PathExprSegment> segments = path->getSegments();
  return resolveSegmentsExpression(resolvedNodeId, segments, offset,
                                   typeSegment, path->getIdentity(),
                                   path->getLocation());
}

TyTy::BaseType *TypeResolver::resolveRootPathExpression(
    std::shared_ptr<ast::PathInExpression> path, size_t *offset,
    basic::NodeId *resolvedNodeId) {

  TyTy::BaseType *rootType = nullptr;
  *offset = 0;

  std::vector<PathExprSegment> segs = path->getSegments();

  NodeId refNodeId = UNKNOWN_NODEID;

  if (auto name = resolver->lookupResolvedName(path->getNodeId())) {
    // llvm::errs() << "resolve root path: it is a name" << "\n";
    refNodeId = *name;
  } else if (auto type = resolver->lookupResolvedType(path->getNodeId())) {
    // llvm::errs() << "resolve root path: it is a type" << "\n";
    refNodeId = *type;
  } else if (auto name = tcx->lookupResolvedName(path->getNodeId())) {
    // llvm::errs() << "tcx:resolve root path: it is a name" << "\n";
    refNodeId = *name;
  } else if (auto type = tcx->lookupResolvedType(path->getNodeId())) {
    // llvm::errs() << "tcx:resolve root path: it is a type" << "\n";
    refNodeId = *type;
  }

  if (refNodeId != UNKNOWN_NODEID) {
    std::optional<TyTy::BaseType *> lookup = queryType(refNodeId);
    if (!lookup) {
      llvm::errs() << "failed to resolve root path3: "
                   << path->getLocation().toString() << "\n";
      return new TyTy::ErrorType(path->getNodeId());
    }
    *offset = 1;
    return *lookup;
  }

  // if (segs.size() == 1)
  //   if (auto t = tcx->lookupBuiltin(segs[0].getIdent().toString())) {
  //     *offset = 1;
  //     return t;
  //   }

  for (unsigned i = 0; i < segs.size(); ++i) {
    PathExprSegment &seg = segs[i];
    bool isRoot = *offset == 0;
    bool haveMoreSegments = segs.size() - 1 != i;
    // NodeId astNodeId = seg.getNodeId();

    NodeId refNodeId = UNKNOWN_NODEID;

    if (auto name = resolver->lookupResolvedName(seg.getNodeId())) {
      refNodeId = *name;
    } else if (auto type = resolver->lookupResolvedType(seg.getNodeId())) {
      refNodeId = *type;
    }

    if (refNodeId == UNKNOWN_NODEID) {
      if (rootType != nullptr && *offset > 0)
        return rootType;

      llvm::errs() << "failed to resolve root segment1: "
                   << seg.getLocation().toString() << "\n";
      llvm::errs() << seg.getNodeId() << "\n";
      return new TyTy::ErrorType(path->getNodeId());
    }

    bool segmentIsModule = tcx->isModule(seg.getNodeId());
    bool segmentIsCrate = tcx->isCrate(seg.getNodeId());

    if (segmentIsModule || segmentIsCrate) {
      if (haveMoreSegments) {
        ++(*offset);
        continue;
      }

      llvm::errs() << "expected value:" << seg.getLocation().toString() << "\n";
      return new TyTy::ErrorType(path->getNodeId());
    }

    std::optional<TyTy::BaseType *> lookup = queryType(seg.getNodeId());
    if (!lookup) {
      if (isRoot) {
        llvm::errs() << "failed to resolve root segment2: "
                     << seg.getLocation().toString() << "\n";
        return new TyTy::ErrorType(path->getNodeId());
      }

      return rootType;
    }

    // enum?
    std::optional<std::pair<ast::Enumeration *, ast::EnumItem *>>
        enumItemLookup = tcx->lookupEnumItem(seg.getNodeId());
    if (enumItemLookup) {
      if ((*enumItemLookup).first != nullptr &&
          (*enumItemLookup).second != nullptr) {
        // it is a legit enum
        tcx->insertVariantDefinition(path->getNodeId(),
                                     (*enumItemLookup).second->getNodeId());
      }
    }

    if (rootType != nullptr) {
      if ((*lookup)->needsGenericSubstitutions())
        if (!rootType->needsGenericSubstitutions()) {
          assert(false);
        }
    }

    if (seg.hasGenerics()) {
      assert(false);
    } else if ((*lookup)->needsGenericSubstitutions()) {
      assert(false);
    }

    *resolvedNodeId = refNodeId;
    *offset = *offset + 1;
    rootType = *lookup;
  }

  return rootType;
}

TyTy::BaseType *TypeResolver::resolveSegmentsExpression(
    basic::NodeId rootResolvedIt, std::span<PathExprSegment> segments,
    size_t offset, TyTy::BaseType *typeSegment, tyctx::NodeIdentity id,
    Location loc) {
  NodeId resolvedNodeId = rootResolvedIt;
  TyTy::BaseType *prevSegment = typeSegment;
  bool receiverIsGeneric = prevSegment->getKind() == TypeKind::Parameter;
  for (size_t i = offset; i < segments.size(); ++i) {
    PathExprSegment &seg = segments[i];
    bool probeImpls = not receiverIsGeneric;

    std::set<PathProbeCandidate> candidates = PathProbeType::probeTypePath(
        prevSegment, seg.getIdent().getIdentifier(), probeImpls,
        false /*probeBounds*/, true /*ignoreMandatoryTraits*/, this);
    if (candidates.size() == 0)
      candidates = PathProbeType::probeTypePath(
          prevSegment, seg.getIdent().getIdentifier(), false,
          true /*probeBounds */, false /*ignoreMandatoryTraits*/, this);
    if (candidates.size() == 0) {
      llvm::errs() << loc.toString()
                   << "@failed to resolve path segment using an impl probe"
                   << "\n";
      exit(EXIT_FAILURE);
    }

    if (candidates.size() > 1) {
      llvm::errs() << "multiple candidates using an impl probe"
                   << "\n";
      exit(EXIT_FAILURE);
    }

    PathProbeCandidate candidate = *candidates.begin();

    prevSegment = typeSegment;
    typeSegment = const_cast<TyTy::BaseType *>(candidate.getType());

    Implementation *associatedImplementation = nullptr;
    if (candidate.isEnumCandidate()) {
      TyTy::VariantDef *variant = candidate.getEnumVariant();

      NodeId variantId = variant->getId();

      std::optional<std::pair<Enumeration *, EnumItem *>> enumItem =
          tcx->lookupEnumItem(variantId);
      if (!enumItem) {
        llvm::errs() << "failed to lookupEnumItem: " << variantId << "\n";
        llvm::errs() << "segment: " << seg.getIdent().toString() << "\n";
        llvm::errs() << "segment: " << seg.getIdent().getLocation().toString()
                     << "\n";
        llvm::errs() << "i: " << i << "\n";
      }
      assert(enumItem.has_value());

      resolvedNodeId = enumItem->second->getNodeId();

      tcx->insertVariantDefinition(id.getNodeId(), variantId);

    } else if (candidate.isImplCandidate()) {
      resolvedNodeId = candidate.getImplNodeId();
      associatedImplementation = candidate.getImplParent();
    } else {
      resolvedNodeId = candidate.getTraitNodeId();

      Implementation *impl = candidate.getTraitImpl();
      if (impl != nullptr) {
        associatedImplementation = impl;
      }
    }

    if (associatedImplementation != nullptr) {
      NodeId implBlockId = associatedImplementation->getNodeId();

      std::optional<AssociatedImplTrait *> associated =
          tcx->lookupAssociatedTraitImpl(implBlockId);
      // assert(associated.has_value());

      auto mappings = TyTy::SubstitutionArgumentMappings::error();
      TyTy::BaseType *implBlockType = resolveImplBlockSelfWithInference(
          associatedImplementation, seg.getLocation(), &mappings);

      if (!mappings.isError()) {
        InternalSubstitutionsMapper mapper;
        typeSegment = mapper.resolve(typeSegment, mappings);
      }

      prevSegment = Unification::unifyWithSite(
          TyTy::WithLocation(prevSegment), TyTy::WithLocation(implBlockType),
          seg.getLocation(), tcx);

      if (prevSegment->getKind() == TyTy::TypeKind::Error)
        return new TyTy::ErrorType(0);

      if (associated) {
        ast::types::TypeExpression *boundPath =
            (*associated)->getTraitImplementation()->getTypePath().get();
        TraitResolver traitResolver;
        TraitReference *traitRef = traitResolver.resolve(boundPath);
        assert(!traitRef->isError());

        TyTy::TypeBoundPredicate predicate =
            implBlockType->lookupPredicate(traitRef->getNodeId());
        if (!predicate.isError())
          implBlockType =
              (*associated)->setupAssociatedTypes(prevSegment, predicate);
      }
    }

    if (typeSegment->needsGenericSubstitutions()) {
      if (!prevSegment->needsGenericSubstitutions()) {
        assert(false);
      }
    }

    if (seg.hasGenerics()) {
      assert(false);
    } else if (typeSegment->needsGenericSubstitutions() && !receiverIsGeneric) {
      Location loc = seg.getLocation();
      SubstitutionsMapper mapper;
      typeSegment = mapper.infer(typeSegment, loc, this);
      if (typeSegment->getKind() == TypeKind::Error)
        exit(EXIT_FAILURE);
    }
  }

  assert(resolvedNodeId != UNKNOWN_NODEID);

  if (typeSegment->needsGenericSubstitutions() && !receiverIsGeneric) {
    Location loc = segments.back().getLocation();
    SubstitutionsMapper mapper;
    typeSegment = mapper.infer(typeSegment, loc, this);
    if (typeSegment->getKind() == TypeKind::Error)
      exit(EXIT_FAILURE);
  }

  tcx->insertReceiver(id.getNodeId(), prevSegment);

  if (resolver->getTypeScope().wasDeclDeclaredInCurrentScope(resolvedNodeId)) {
    resolver->insertResolvedType(id.getNodeId(), resolvedNodeId);
  } else {
    resolver->insertResolvedMisc(id.getNodeId(), resolvedNodeId);
  }

  return typeSegment;
}

TyTy::BaseType *
TypeResolver::resolveImplBlockSelf(const AssociatedImplTrait &) {
  assert(false);
}

TyTy::BaseType *TypeResolver::resolveImplBlockSelfWithInference(
    ast::Implementation *impl, Location loc,
    TyTy::SubstitutionArgumentMappings *inferArguments) {

  bool failedFlag = false;
  std::vector<TyTy::SubstitutionParamMapping> substitutions =
      resolveImplBlockSubstitutions(impl, failedFlag);

  if (failedFlag)
    return new TyTy::ErrorType(impl->getNodeId());

  TyTy::BaseType *self = resolveImplBlockSelf(impl);
  if (substitutions.empty() || self->isConcrete())
    return self;

  std::vector<TyTy::SubstitutionArg> args;
  for (auto &p : substitutions) {
    if (p.needSubstitution()) {
      TyTy::TypeVariable inferVar =
          TyTy::TypeVariable::getImplicitInferVariable(loc);
      args.push_back(TyTy::SubstitutionArg(&p, inferVar.getType()));
    } else {
      TyTy::ParamType *param = p.getParamType();
      TyTy::BaseType *resolved = param->destructure();
      args.push_back(TyTy::SubstitutionArg(&p, resolved));
    }
  }

  *inferArguments =
      TyTy::SubstitutionArgumentMappings(std::move(args), {}, loc);

  InternalSubstitutionsMapper intern;

  TyTy::BaseType *infer = intern.resolve(self, *inferArguments);

  if (!infer->hasSubsititionsDefined()) {
    for (auto &bound : infer->getSpecifiedBounds()) {
      bound.handleSubstitions(*inferArguments);
    }
  }

  return infer;
}

TyTy::BaseType *TypeResolver::resolveImplBlockSelf(const Implementation *impl) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    return checkType(static_cast<const InherentImpl *>(impl)->getType());
  }
  case ImplementationKind::TraitImpl: {
    return checkType(static_cast<const TraitImpl *>(impl)->getType());
  }
  }
}

std::vector<TyTy::SubstitutionParamMapping>
TypeResolver::resolveImplBlockSubstitutions(ast::Implementation *impl,
                                            bool &failedFlag) {
  switch (impl->getKind()) {
  case ImplementationKind::InherentImpl: {
    return resolveImplBlockSubstitutions(static_cast<InherentImpl *>(impl),
                                         failedFlag);
  }
  case ImplementationKind::TraitImpl: {
    return resolveImplBlockSubstitutions(static_cast<TraitImpl *>(impl),
                                         failedFlag);
  }
  }
}

std::vector<TyTy::SubstitutionParamMapping>
TypeResolver::resolveImplBlockSubstitutions(ast::InherentImpl *impl,
                                            bool &failedFlag) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (impl->hasGenericParams())
    checkGenericParams(impl->getGenericParams(), substitutions);

  if (impl->hasWhereClause())
    checkWhereClause(impl->getWhereClause());

  TyTy::TypeBoundPredicate specifiedBounds = TyTy::TypeBoundPredicate::error();

  TyTy::BaseType *self = checkType(impl->getType());

  if (!specifiedBounds.isError())
    self->inheritBounds({specifiedBounds});

  TyTy::SubstitutionArgumentMappings traitConstraints =
      specifiedBounds.getSubstitutionArguments();
  TyTy::SubstitutionArgumentMappings implConstraints =
      getUsedSubstitutionArguments(self);

  failedFlag = checkForUnconstrained(substitutions, traitConstraints,
                                     implConstraints, self);

  return substitutions;
}

std::vector<TyTy::SubstitutionParamMapping>
TypeResolver::resolveImplBlockSubstitutions(ast::TraitImpl *impl,
                                            bool &failedFlag) {
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  if (impl->hasGenericParams())
    checkGenericParams(impl->getGenericParams(), substitutions);

  if (impl->hasWhereClause())
    checkWhereClause(impl->getWhereClause());

  TyTy::TypeBoundPredicate specifiedBounds = TyTy::TypeBoundPredicate::error();

  TraitReference *traitReference = &TraitReference::errorNode();

  TraitResolver traitResolver;
  traitReference = traitResolver.resolve(impl->getTypePath().get());
  assert(!traitReference->isError());

  specifiedBounds =
      getPredicateFromBound(impl->getTypePath(), impl->getType().get());

  TyTy::BaseType *self = checkType(impl->getType());

  if (!specifiedBounds.isError())
    self->inheritBounds({specifiedBounds});

  TyTy::SubstitutionArgumentMappings traitConstraints =
      specifiedBounds.getSubstitutionArguments();
  TyTy::SubstitutionArgumentMappings implConstraints =
      getUsedSubstitutionArguments(self);

  failedFlag = checkForUnconstrained(substitutions, traitConstraints,
                                     implConstraints, self);

  return substitutions;
}

} // namespace rust_compiler::sema::type_checking
