#include "AST/Statement.h"

#include "AST/ExpressionStatement.h"
#include "CrateBuilder/CrateBuilder.h"

#include <memory>

using namespace rust_compiler::ast;

namespace rust_compiler::crate_builder {

void CrateBuilder::emitStatement(ast::Statement *stmt) {
  switch (stmt->getKind()) {
  case StatementKind::EmptyStatement: {
    // empty
    break;
  }
  case ast::StatementKind::ItemDeclaration: {
    break;
  }
  case ast::StatementKind::LetStatement: {
    emitLetStatement(static_cast<ast::LetStatement *>(stmt));
    break;
  }
  case ast::StatementKind::ExpressionStatement: {
    emitExpressionStatement(static_cast<ast::ExpressionStatement *>(stmt));
    break;
  }
  case ast::StatementKind::MacroInvocationSemi: {
    break;
  }
  }
}

void CrateBuilder::emitLetStatement(ast::LetStatement *let) {
  if (let->hasInit()) {
    std::optional<mlir::Value> init = emitExpression(let->getInit().get());
    if (init) {
    }
  }
  assert(false);
}

} // namespace rust_compiler::crate_builder
