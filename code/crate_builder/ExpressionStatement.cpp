#include "AST/ExpressionStatement.h"

#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

void CrateBuilder::emitExpressionStatement(
    std::shared_ptr<ast::ExpressionStatement> stmt) {
  // FIXME
  assert(false);

  switch (stmt->getKind()) {
  case ast::ExpressionStatementKind::ExpressionWithoutBlock: {
    break;
  }
  case ast::ExpressionStatementKind::ExpressionWithBlock: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
