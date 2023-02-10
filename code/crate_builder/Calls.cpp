#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Value
CrateBuilder::emitCallExpression(std::shared_ptr<ast::CallExpression> expr) {}

mlir::Value
emitMethodCallExpression(std::shared_ptr<ast::MethodCallExpression> expr) {}

} // namespace rust_compiler::crate_builder
