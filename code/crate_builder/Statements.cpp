#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitStatements(ast::Statements stmts) {

  for (auto& stmt: stmts.getStmts())
    emitStatement(stmt);

  if (stmts.hasTrailing())
    return emitExpressionWithoutBlock(stmts.getTrailing());

  return Unit;
}

} // namespace rust_compiler::crate_builder
