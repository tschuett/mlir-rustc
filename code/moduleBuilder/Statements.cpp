#include "AST/Statement.h"
#include "ModuleBuilder.h"

namespace rust_compiler {

void ModuleBuilder::buildStatement(std::shared_ptr<ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::Item:
  case ast::StatementKind::LetStatement:
  case ast::StatementKind::MacroInvocationSemi:
  }
}

void ModuleBuilder::buildLetStatement(
    std::shared_ptr<ast::LetStatement> letStmt) {}

} // namespace rust_compiler
