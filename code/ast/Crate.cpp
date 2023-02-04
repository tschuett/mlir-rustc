#include "AST/Crate.h"

namespace rust_compiler::ast {

void Crate::merge(std::shared_ptr<ast::Module> module,
                  adt::CanonicalPath path) {
  assert(false);
}

std::span<std::shared_ptr<Item>> Crate::getItems() const {
  assert(false); //FIXME
}

} // namespace rust_compiler::ast
