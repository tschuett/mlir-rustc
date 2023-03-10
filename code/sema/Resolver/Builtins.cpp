#include "Resolver.h"

using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

void Resolver::generateBuiltins() {
  assert(false && "to be handled later");

  auto u8 = new TyTy::UintType(mappings->getNextNodeId(), TyTy::UintKind::U8);

  setupBuiltin("u8", u8);
}

void Resolver::setupBuiltin(std::string_view name, TyTy::BaseType *tyty) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
