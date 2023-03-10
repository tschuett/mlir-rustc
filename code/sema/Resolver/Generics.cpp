#include "Resolver.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::adt;

namespace rust_compiler::sema::resolver {

void Resolver::resolveWhereClause(const WhereClause &) {
  // FIXME
  assert(false && "to be handled later");
}

void Resolver::resolveGenericParams(const GenericParams &,
                                    const adt::CanonicalPath &prefix,
                                    const adt::CanonicalPath &canonicalPrefix) {
  // FIXME
  assert(false && "to be handled later");
}

void Resolver::resolveGenericArgs(const ast::GenericArgs &) {
  assert(false && "to be handled later");
}

} // namespace rust_compiler::sema::resolver
