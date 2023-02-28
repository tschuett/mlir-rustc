#include "AST/Crate.h"

namespace rust_compiler::ast {

void Crate::merge(std::shared_ptr<ast::Module> module,
                  adt::CanonicalPath path) {
  assert(false);
}

std::string_view Crate::getCrateName() const { return crateName; }

} // namespace rust_compiler::ast
