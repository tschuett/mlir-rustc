#include "CrateBuilder/CrateBuilder.h"

#include <memory>

namespace rust_compiler::crate_builder {

void CrateBuilder::emitFunction(std::shared_ptr<ast::Function> f) {
  assert(false);
  if (f->hasBody())
    emitBlockExpression(
                        std::static_pointer_cast<ast::BlockExpression>(f->getBody()));
}

} // namespace rust_compiler::crate_builder
