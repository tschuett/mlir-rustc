#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

void CrateBuilder::emitFunction(std::shared_ptr<ast::Function> f) {
  assert(false);
  if (f->hasBody())
    emitBlockExpression(f->getBody());
}

} // namespace rust_compiler::crate_builder
