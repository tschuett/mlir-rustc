#include "ModuleBuilder/ModuleBuilder.h"

#include <optional>

namespace rust_compiler {

mlir::Value ModuleBuilder::buildExpressionWithoutBlock(
    std::shared_ptr<ast::ExpressionWithoutBlock> expr) {
  // FIXME
  return nullptr;
}

} // namespace rust_compiler
