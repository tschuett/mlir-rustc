#include "AST/PathExpression.h"

#include "AST/PathInExpression.h"
#include "AST/QualifiedPathInExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <cassert>
#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value
ModuleBuilder::emitPathExpression(std::shared_ptr<ast::PathExpression> path) {

  switch (path->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression: {
    return emitPathInExpression(
        std::static_pointer_cast<ast::PathInExpression>(path));
  }
  case PathExpressionKind::QualifiedPathInExpression: {
    return emitQualifiedPathInExpression(
        std::static_pointer_cast<ast::QualifiedPathInExpression>(path));
  }
  }

  assert(false);
}

mlir::Value ModuleBuilder::emitPathInExpression(
    std::shared_ptr<ast::PathInExpression> path) {
  std::vector<PathExprSegment> segments = path->getSegments();
  assert(segments.size() == 1);

  llvm::outs() << segments[0].getIdent() << "\n";

  if (symbolTable.count(segments[0].getIdent()) == 1) {
    return std::get<0>(symbolTable.lookup(segments[0].getIdent()));
  }

  assert(false);
}

mlir::Value ModuleBuilder::emitQualifiedPathInExpression(
    std::shared_ptr<ast::QualifiedPathInExpression> path) {
  // symbolTable.
  assert(false);
}

} // namespace rust_compiler
