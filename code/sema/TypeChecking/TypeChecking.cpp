#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

TypeCheckContext *TypeCheckContext::get() {
  static TypeCheckContext *instance;
  if (instance == nullptr)
    instance = new TypeCheckContext();
  return instance;
}

void TypeCheckContext::checkCrate(std::shared_ptr<ast::Crate>) {
  assert(false && "to be done");
}

} // namespace rust_compiler::sema::type_checking
