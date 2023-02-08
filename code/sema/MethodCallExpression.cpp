#include "AST/PathIdentSegment.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

static bool isIdentifier(const PathExprSegment &path) {
  if (not path.hasGenerics()) {
    PathIdentSegment ident = path.getIdent();
    return ident.getKind() == PathIdentSegmentKind::Identifier;
  }
  return false;
}

void Sema::analyzeMethodCallExpression(
    std::shared_ptr<ast::MethodCallExpression> let) {
  PathExprSegment path = let->getPath();

  if (isIdentifier(path)) {
  }
}

} // namespace rust_compiler::sema

/*
  .unwarp()
  .clone()
  .to_string()
 */
