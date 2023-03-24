#pragma once

#include "AST/PathIdentSegment.h"
#include "TyTy.h"

#include <set>

namespace rust_compiler::sema::type_checking {

class PathProbeCandidate {};

std::set<PathProbeCandidate> probeTypePath(TyTy::BaseType *receiver,
                                           ast::PathIdentSegment segment,
                                           bool probeImpls, bool probeBounds,
                                           bool ignoreTraitItems);

} // namespace rust_compiler::sema::type_checking
