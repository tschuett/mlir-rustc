#include "AST/PathIdentSegment.h"
#include "TypeChecking.h"

namespace rust_compiler::sema::type_checking {

std::set<MethodCandidate>
TypeResolver::probeMethodResolver(TyTy::BaseType *receiver,
                                  const ast::PathIdentSegment &segmentName,
                                  bool autoDeref) {
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
