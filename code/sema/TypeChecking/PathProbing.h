#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Implementation.h"
#include "AST/PathIdentSegment.h"
#include "Sema/Autoderef.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"

#include <set>
#include <variant>

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
  struct EnumItem {
    TyTy::ADTType *parent;
    TyTy::VariantDef *variant;
  };
  struct ImplItem {
    ast::AssociatedItem *implItem;
    ast::Implementation *parent;
  };
  struct TraitItem {
    TraitReference *traitRef;
    TraitItemReference *itemRef;
    ast::Implementation *impl;
  };

public:
  bool isImplCandidate() const {
    return std::holds_alternative<ImplItem>(candidate);
  }
  CandidateKind getKind() const;
  TyTy::BaseType *getType() const { return type; }

  basic::NodeId getImplNodeId() const {
    return std::get<ImplItem>(candidate).implItem->getNodeId();
  }
  basic::NodeId getTraitNodeId() const {
    return std::get<TraitItem>(candidate).itemRef->getNodeId();
  }

private:
  TyTy::BaseType *type;

  std::variant<EnumItem, ImplItem, TraitItem> candidate;
};

class MethodCandidate {
  PathProbeCandidate candidate;
  std::vector<sema::Adjustment> adjustments;

public:
  std::vector<sema::Adjustment> getAdjustments() const { return adjustments; }
  PathProbeCandidate getCandidate() const { return candidate; }
};

std::set<PathProbeCandidate> probeTypePath(TyTy::BaseType *receiver,
                                           ast::PathIdentSegment segment,
                                           bool probeImpls, bool probeBounds,
                                           bool ignoreTraitItems);

} // namespace rust_compiler::sema::type_checking
