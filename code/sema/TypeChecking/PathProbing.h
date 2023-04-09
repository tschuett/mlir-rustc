#pragma once

#include "AST/PathIdentSegment.h"
#include "TyCtx/TyTy.h"

#include <set>

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::tyctx;

class PathProbeCandidate {};

std::set<PathProbeCandidate> probeTypePath(TyTy::BaseType *receiver,
                                           ast::PathIdentSegment segment,
                                           bool probeImpls, bool probeBounds,
                                           bool ignoreTraitItems);

} // namespace rust_compiler::sema::type_checking
