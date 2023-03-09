#include "Resolver.h"

namespace rust_compiler::sema::resolver {

void Resolver::resolvePatternDeclaration(
    std::shared_ptr<ast::patterns::PatternNoTopAlt>, Rib::RibKind) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
