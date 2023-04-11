#include "AST/Expression.h"
#include "CrateBuilder/CrateBuilder.h"


namespace rust_compiler::crate_builder {

std::optional<mlir::Value> CrateBuilder::emitStatements(ast::Statements stmts) {

  for (auto &stmt : stmts.getStmts())
    emitStatement(stmt.get());

  if (stmts.hasTrailing())
    return emitExpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(stmts.getTrailing().get()));

  return std::nullopt;
}

} // namespace rust_compiler::crate_builder
