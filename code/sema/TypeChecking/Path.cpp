#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "Basic/Ids.h"
#include "TyTy.h"
#include "TypeChecking.h"

#include "../Resolver/Resolver.h"

#include <memory>

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

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

  size_t offset = 1;
  NodeId resolvedNodeId = UNKNOWN_NODEID;

  TyTy::BaseType *typeSegment =
      resolveRootPathExpression(path, &offset, &resolvedNodeId);
  if (typeSegment->getKind() == TyTy::TypeKind::Error) {
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

  std::vector<PathExprSegment> segs = path->getSegments();

  llvm::errs() << "resolveRootPathExpr: " << segs[0].getIdent().toString()
               << "\n";

  if (segs.size() == 1)
    if (auto t = tcx->lookupBuiltin(segs[0].getIdent().toString())) {
      *offset = 1;
      return t;
    }

  NodeId refNodeId = UNKNOWN_NODEID;
  for (unsigned i = 0; i < segs.size(); ++i) {
    PathExprSegment &seg = segs[0];
    bool isRoot = *offset == 0;
    bool haveMoreSegments = segs.size() - 1 != i;
    NodeId astNodeId = seg.getNodeId();

    NodeId refNodeId = UNKNOWN_NODEID;

    if (auto name = resolver->lookupResolvedName(seg.getNodeId())) {
      refNodeId = *name;
    } else if (auto type = resolver->lookupResolvedType(seg.getNodeId())) {
      refNodeId = *type;
    }

    if (refNodeId == UNKNOWN_NODEID) {
      if (rootType != nullptr && *offset > 0)
        return rootType;

      llvm::errs() << "failed to resolve root segment"
                   << "\n";
      return new TyTy::ErrorType(path->getNodeId());
    }

    assert(false && "to be implemented");
  }
}

TyTy::BaseType *TypeResolver::resolveSegmentsExpression(
    basic::NodeId rootResolvedIt, std::span<PathExprSegment> segment,
    size_t offset, TyTy::BaseType *typeSegment, tyctx::NodeIdentity, Location) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
