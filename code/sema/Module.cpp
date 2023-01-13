#include "AST/Module.h"

#include "Sema/Sema.h"

namespace rust_compiler::sema {

void Sema::walkModule(std::shared_ptr<ast::Module> module) {
  for (auto &item : module->getItems()) {
    walkItem(item);
  }
}

} // namespace rust_compiler::sema

// FIXME: ModuleTree
