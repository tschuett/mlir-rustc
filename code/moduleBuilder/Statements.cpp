#include "AST/ExpressionStatement.h"
#include "AST/ItemDeclaration.h"
#include "AST/Statement.h"
#include "ModuleBuilder/ModuleBuilder.h"

#include <optional>

namespace rust_compiler {

std::optional<mlir::Value> ModuleBuilder::emitStatement(
    std::shared_ptr<rust_compiler::ast::Statement> stmt) {
  llvm::outs() << "emitStatement"
               << "\n";

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

void ModuleBuilder::buildExpressionStatement(
    std::shared_ptr<ast::ExpressionStatement> expr) {}

void ModuleBuilder::emitItemDeclaration(
    std::shared_ptr<ast::ItemDeclaration> item) {}

std::optional<mlir::Value>
ModuleBuilder::emitStatements(std::shared_ptr<ast::Statements> stmts) {

  llvm::outs() << "emitStatements: " << stmts->getStmts().size() << " "
               << stmts->hasTrailing() << "\n";

  for (auto &stmt : stmts->getStmts()) {
    emitStatement(stmt);
  }

  if (stmts->hasTrailing())
    return emitExpressionWithoutBlock(
        std::static_pointer_cast<ast::ExpressionWithoutBlock>(
            stmts->getTrailing()));

  return std::nullopt;
}

} // namespace rust_compiler
