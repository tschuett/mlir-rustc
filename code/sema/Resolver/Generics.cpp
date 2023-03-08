#include "Resolver.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveWhereClause(const WhereClause &) {
  // FIXME
}

void Resolver::resolveGenericParams(const GenericParams &,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix) {
  // FIXME
}

} // namespace rust_compiler::sema::resolver
