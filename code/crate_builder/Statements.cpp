#include "CrateBuilder/CrateBuilder.h"
#include "Hir/HirOps.h"

using namespace rust_compiler::hir;

namespace rust_compiler::crate_builder {

std::optional<mlir::Value> CrateBuilder::emitStatements(ast::Statements stmts) {

  for (auto &stmt : stmts.getStmts())
    emitStatement(stmt);

  if (stmts.hasTrailing())
    return emitExpressionWithoutBlock(stmts.getTrailing());

  return std::nullopt;
}

} // namespace rust_compiler::crate_builder
