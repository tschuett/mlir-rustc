#include "Resolver.h"

namespace rust_compiler::sema::resolver {

void Resolver::resolveFunction(std::shared_ptr<ast::Function>,
                               const adt::CanonicalPath &prefix,
                               const adt::CanonicalPath &canonicalPrefix) {}

} // namespace rust_compiler::sema::resolver
