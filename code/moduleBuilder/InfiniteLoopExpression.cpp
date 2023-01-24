#include "ModuleBuilder/ModuleBuilder.h"

namespace rust_compiler {

mlir::Value ModuleBuilder::emitInfiniteLoopExpression(
    std::shared_ptr<ast::InfiniteLoopExpression> infi) {

  assert(false);

  infi->getBody()->containsBreakExpression();
}

} // namespace rust_compiler
