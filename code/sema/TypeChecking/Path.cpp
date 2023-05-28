#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "Basic/Ids.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

#include "../Resolver/Resolver.h"

#include <cstddef>
#include <llvm/Support/raw_ostream.h>
#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;
using namespace rust_compiler::tyctx;

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
    //llvm::errs() << "resolve root path: it is a name" << "\n";
    refNodeId = *name;
  } else if (auto type = resolver->lookupResolvedType(path->getNodeId())) {
    //llvm::errs() << "resolve root path: it is a type" << "\n";
    refNodeId = *type;
  } else if (auto name = tcx->lookupResolvedName(path->getNodeId())) {
    //llvm::errs() << "tcx:resolve root path: it is a name" << "\n";
    refNodeId = *name;
  } else if (auto type = tcx->lookupResolvedType(path->getNodeId())) {
    //llvm::errs() << "tcx:resolve root path: it is a type" << "\n";
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

    assert(false && "to be implemented");
  }
  assert(false && "to be implemented");
}

TyTy::BaseType *TypeResolver::resolveSegmentsExpression(
    basic::NodeId rootResolvedIt, std::span<PathExprSegment> segment,
    size_t offset, TyTy::BaseType *typeSegment, tyctx::NodeIdentity, Location) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
