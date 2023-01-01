#include "AST/Statement.h"
#include "ModuleBuilder/ModuleBuilder.h"

namespace rust_compiler {

void ModuleBuilder::buildStatement(std::shared_ptr<ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::Item: {
    break;
  }
  case ast::StatementKind::LetStatement: {
    break;
  }
  case ast::StatementKind::MacroInvocationSemi: {
    break;
  }
  case ast::StatementKind::ExpressionStatement: {
    break;
  }
  }
}

void ModuleBuilder::buildLetStatement(
    std::shared_ptr<ast::LetStatement> letStmt) {}

} // namespace rust_compiler
