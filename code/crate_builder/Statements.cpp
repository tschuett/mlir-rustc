#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitStatements(ast::Statements stmts) {

  for (auto& stmt: stmts.getStmts())
    emitStatement(stmt);

  if (stmts.hasTrailing())
    emitExpressionWithoutBlock(stmts.getTrailing());
}

} // namespace rust_compiler::crate_builder
