#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "TypeChecking.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkPathExpression(std::shared_ptr<ast::PathExpression> path) {
  assert(false && "to be implemented");
  switch (path->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression: {
    return checkPathInExpression(
        std::static_pointer_cast<PathInExpression>(path));
  }
  case PathExpressionKind::QualifiedPathInExpression: {
  }
  }
}

TyTy::BaseType *TypeResolver::checkPathInExpression(
    std::shared_ptr<ast::PathInExpression> path) {
  assert(false && "to be implemented");

  std::vector<PathExprSegment> segments = path->getSegments();
}

} // namespace rust_compiler::sema::type_checking
