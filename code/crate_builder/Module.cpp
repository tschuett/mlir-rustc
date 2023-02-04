#include "CrateBuilder/CrateBuilder.h"

#include <memory>

namespace rust_compiler::crate_builder {

void CrateBuilder::emitModule(std::shared_ptr<ast::Module> module) {

  for (auto &mod : module->getItems()) {
    emitItem(mod);
  }
}

} // namespace rust_compiler::crate_builder
