#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Type CrateBuilder::getType(ast::types::TypeExpression *) {
  assert(false);
}

} // namespace rust_compiler::crate_builder
