#include "Resolver.h"

#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <optional>

using namespace rust_compiler::basic;
using namespace rust_compiler::adt;
using namespace rust_compiler::ast;
using namespace rust_compiler::sema::type_checking;

namespace rust_compiler::sema::resolver {

void Resolver::resolveMatchExpression(
    std::shared_ptr<ast::MatchExpression>, const adt::CanonicalPath &prefix,
    const adt::CanonicalPath &canonicalPrefix) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
