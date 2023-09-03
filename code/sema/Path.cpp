#include "AST/PathExpression.h"
#include "AST/PathInExpression.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

using namespace rust_compiler::sema;

void Sema::analyzePathExpression(ast::PathExpression *path) {
  switch (path->getPathExpressionKind()) {
  case PathExpressionKind::PathInExpression:
    analyzePathInExpression(static_cast<PathInExpression *>(path));
    break;
  case PathExpressionKind::QualifiedPathInExpression:
    assert(false);
  }
}

void Sema::analyzePathInExpression(ast::PathInExpression *) {}
