#include "AST/ExpressionStatement.h"

#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

void CrateBuilder::emitExpressionStatement(ast::ExpressionStatement *stmt) {
  switch (stmt->getKind()) {
  case ast::ExpressionStatementKind::ExpressionWithoutBlock: {
    emitExpressionWithoutBlock(static_cast<ast::ExpressionWithoutBlock *>(
        stmt->getWithoutBlock().get()));
    break;
  }
  case ast::ExpressionStatementKind::ExpressionWithBlock: {
    emitExpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(stmt->getWithBlock().get()));
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
