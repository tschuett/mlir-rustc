#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

TypeCheckContext *TypeCheckContext::get() {
  static TypeCheckContext *instance;
  if (instance == nullptr)
    instance = new TypeCheckContext();

  return instance;
}

} // namespace rust_compiler::sema::type_checking
