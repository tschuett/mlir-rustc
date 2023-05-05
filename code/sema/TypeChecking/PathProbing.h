#pragma once

#include "AST/PathIdentSegment.h"
#include "Sema/Autoderef.h"
#include "TyCtx/TyTy.h"

#include <set>

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::tyctx;

enum class CandidateKind {
  EnumVariant,

  ImplConst,
  ImplTypeAlias,
  ImplFunc,

  TraitItemConst,
  TraitTypeAlias,
  TraitFunc,

  Error

};

class PathProbeCandidate {
public:
  bool isImplCandidate() const;
  CandidateKind getKind() const;
  TyTy::BaseType *getType() const { return type; }

  basic::NodeId getImplNodeId() const;
  basic::NodeId getTraitNodeId() const;
private:
  TyTy::BaseType *type;
};

class MethodCandidate {
  // FIXME
  std::vector<sema::Adjustment> adjustments;

public:
  std::vector<sema::Adjustment> getAdjustments() const { return adjustments; }
  PathProbeCandidate getCandidate();
};

std::set<PathProbeCandidate> probeTypePath(TyTy::BaseType *receiver,
                                           ast::PathIdentSegment segment,
                                           bool probeImpls, bool probeBounds,
                                           bool ignoreTraitItems);

} // namespace rust_compiler::sema::type_checking
