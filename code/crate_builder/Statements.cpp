#include "CrateBuilder/CrateBuilder.h"

#include "Hir/HirOps.h"

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitStatements(ast::Statements stmts) {

  for (auto& stmt: stmts.getStmts())
    emitStatement(stmt);

  if (stmts.hasTrailing())
    return emitExpressionWithoutBlock(stmts.getTrailing());

  return builder.create<UnitConstantOp>();
}

} // namespace rust_compiler::crate_builder
