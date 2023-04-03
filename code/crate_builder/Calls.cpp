#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitCallExpression(std::shared_ptr<ast::CallExpression> expr) {
  assert(false);
}

mlir::Value
emitMethodCallExpression(std::shared_ptr<ast::MethodCallExpression> expr) {
  assert(false);
}

} // namespace rust_compiler::crate_builder
