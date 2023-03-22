#include "PathProbing.h"

namespace rust_compiler::sema::type_checking {

std::set<PathProbeCandidate> probeTypePath(TyTy::BaseType *receiver,
                                           ast::PathIdentSegment segment,
                                           bool probeImpls, bool probeBounds,
                                           bool ignoreTraitItems) {
  assert(false && "to be implemented");
}

} // namespace rust_compiler::sema::type_checking
