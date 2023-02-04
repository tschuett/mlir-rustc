#include "AST/Statement.h"

#include "AST/ExpressionStatement.h"
#include "CrateBuilder/CrateBuilder.h"
#include <memory>

namespace rust_compiler::crate_builder {

void CrateBuilder::emitStatement(std::shared_ptr<ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::ItemDeclaration: {
    break;
  }
  case ast::StatementKind::LetStatement: {
    break;
  }
  case ast::StatementKind::ExpressionStatement: {
    emitExpressionStatement(std::static_pointer_cast<ast::ExpressionStatement>(stmt));
    break;
  }
  case ast::StatementKind::MacroInvocationSemi: {
    break;
  }
  }
}

} // namespace rust_compiler::crate_builder
