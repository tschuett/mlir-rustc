#include "AST/AssociatedItem.h"
#include "AST/EnumItem.h"
#include "AST/Implementation.h"
#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "Basic/Ids.h"
#include "PathProbing.h"
#include "TyCtx/SubstitutionsMapper.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"
#include "Unification.h"

#include "../Resolver/Resolver.h"

#include <cstddef>
#include <cstdlib>
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

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
    Location) {
  NodeId resolvedNodeId = rootResolvedIt;
  TyTy::BaseType *prevSegment = typeSegment;
  bool receiverIsGeneric = prevSegment->getKind() == TypeKind::Parameter;
  for (size_t i = offset; i < segments.size(); ++i) {
    PathExprSegment &seg = segments[i];
    bool probeImpls = not receiverIsGeneric;

    std::set<PathProbeCandidate> candidates = PathProbeType::probeTypePath(
        prevSegment, seg.getIdent().getIdentifier(), probeImpls, false, true,
        this);
    if (candidates.size() == 0)
      candidates = PathProbeType::probeTypePath(prevSegment,
                                                seg.getIdent().getIdentifier(),
                                                false, true, false, this);
    if (candidates.size() == 0) {
      llvm::errs() << "failed to resolve path segment using an impl probe"
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
      assert(associated.has_value());
      TyTy::BaseType *implBlockType = resolveImplBlockSelf(*(*associated));
      if (implBlockType->needsGenericSubstitutions()) {
        SubstitutionsMapper mapper;
        implBlockType = mapper.infer(implBlockType, seg.getLocation(), this);
      }

      prevSegment = Unification::unifyWithSite(
          TyTy::WithLocation(prevSegment), TyTy::WithLocation(implBlockType),
          seg.getLocation(), tcx);
      if (prevSegment->getKind() == TypeKind::Error) {
        llvm::errs() << "resolveSegmentsExpression: unification failed"
                     << "\n";
        exit(EXIT_FAILURE);
      }

      if (associated) {
        TraitImpl *impl = (*associated)->getImplementation();

        std::shared_ptr<ast::types::TypeExpression> boundPath =
            impl->getTypePath();

        NodeId implicitId = rust_compiler::basic::getNextNodeId();
        tcx->insertImplicitType(implicitId, implBlockType);

        ast::types::TypeExpression *implicitSelfBound =
            new ast::types::TypePath(Location::getEmptyLocation());

        TyTy::TypeBoundPredicate predicate =
            getPredicateFromBound(boundPath, implicitSelfBound);
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

} // namespace rust_compiler::sema::type_checking
