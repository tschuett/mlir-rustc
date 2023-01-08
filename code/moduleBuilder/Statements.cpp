#include "AST/ExpressionStatement.h"
#include "AST/ItemDeclaration.h"
#include "AST/Statement.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <optional>

namespace rust_compiler {

std::optional<mlir::Value> ModuleBuilder::emitStatement(
    std::shared_ptr<rust_compiler::ast::Statement> stmt) {
  switch (stmt->getKind()) {
  case ast::StatementKind::ItemDeclaration: {
    emitItemDeclaration(
        static_pointer_cast<rust_compiler::ast::ItemDeclaration>(stmt));
    return std::nullopt;
  }
  case ast::StatementKind::LetStatement: {
    buildLetStatement(
        static_pointer_cast<rust_compiler::ast::LetStatement>(stmt));
    return std::nullopt;
  }
  case ast::StatementKind::MacroInvocationSemi: {
    // FIXME
    return std::nullopt;
    break;
  }
  case ast::StatementKind::ExpressionStatement: {
    buildExpressionStatement(
        static_pointer_cast<rust_compiler::ast::ExpressionStatement>(stmt));
    return std::nullopt;
  }
  }
}

void ModuleBuilder::buildLetStatement(
    std::shared_ptr<ast::LetStatement> letStmt) {}

void ModuleBuilder::buildExpressionStatement(
    std::shared_ptr<ast::ExpressionStatement> expr) {}

void ModuleBuilder::emitItemDeclaration(
    std::shared_ptr<ast::ItemDeclaration> item) {}

} // namespace rust_compiler
