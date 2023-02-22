#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

std::optional<mlir::Value>
CrateBuilder::emitBlockExpression(std::shared_ptr<ast::BlockExpression> block) {
  return emitStatements(block->getExpressions());
}

} // namespace rust_compiler::crate_builder
