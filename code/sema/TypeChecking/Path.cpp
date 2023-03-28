#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "Basic/Ids.h"
#include "TyTy.h"
#include "TypeChecking.h"

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
  assert(false && "to be implemented");

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
  return resolveSegmentsExpression(resolvedNodeId, segments, offset, typeSegment,
                                  path->getIdentity(), path->getLocation());
}

TyTy::BaseType *TypeResolver::resolveSegmentsExpression(
    basic::NodeId rootResolvedIt, std::span<PathExprSegment> segment,
    size_t offset, TyTy::BaseType *typeSegment, tyctx::NodeIdentity, Location) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
