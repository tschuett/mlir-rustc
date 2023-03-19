#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

void TypeResolver::checkPattern(
    std::shared_ptr<ast::patterns::PatternNoTopAlt> pat, TyTy::BaseType *t) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
