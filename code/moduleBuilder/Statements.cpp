#include "AST/ExpressionStatement.h"
#include "AST/Statement.h"
#include "ModuleBuilder/ModuleBuilder.h"

namespace rust_compiler {

void ModuleBuilder::buildStatement(
    std::shared_ptr<rust_compiler::ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::Item: {
    buildItem(static_pointer_cast<rust_compiler::ast::Item>(stmt));
    break;
  }
  case ast::StatementKind::LetStatement: {
    buildLetStatement(
        static_pointer_cast<rust_compiler::ast::LetStatement>(stmt));
    break;
  }
  case ast::StatementKind::MacroInvocationSemi: {
    break;
  }
  case ast::StatementKind::ExpressionStatement: {
    buildExpressionStatement(
        static_pointer_cast<rust_compiler::ast::ExpressionStatement>(stmt));
    break;
  }
  }
}

void ModuleBuilder::buildLetStatement(
    std::shared_ptr<ast::LetStatement> letStmt) {}

void ModuleBuilder::buildExpressionStatement(
    std::shared_ptr<ast::ExpressionStatement> expr) {}

} // namespace rust_compiler
