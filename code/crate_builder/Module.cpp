#include "CrateBuilder/CrateBuilder.h"

#include <memory>

namespace rust_compiler::crate_builder {

void CrateBuilder::emitModule(ast::Module *module) {
  for (auto &mod : module->getItems())
    emitItem(mod.get());
}

} // namespace rust_compiler::crate_builder
