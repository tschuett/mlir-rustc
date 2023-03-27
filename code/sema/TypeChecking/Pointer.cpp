#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *TypeResolver::checkEnumerationPointer(ast::Enumeration *e) {
  assert(false && "to be implemented");
}

TyTy::BaseType *
TypeResolver::checkImplementationPointer(ast::Implementation *i) {
  assert(false && "to be implemented");
}

TyTy::BaseType *TypeResolver::checkExternalItemPointer(ast::ExternalItem *e) {
  assert(false && "to be implemented");
}

TyTy::BaseType *TypeResolver::checkItemPointer(ast::Item *e) {
  assert(false && "to be implemented");
}

TyTy::BaseType *
TypeResolver::checkAssociatedItemPointer(ast::AssociatedItem *,
                                         ast::Implementation *) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
