#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Implementation.h"
#include "AST/PathIdentSegment.h"
#include "Sema/Autoderef.h"
#include "Session/Session.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"

#include <set>
#include <variant>

namespace rust_compiler::sema::type_checking {

using namespace rust_compiler::tyctx;
using namespace rust_compiler::adt;

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
  struct EnumItem {
    TyTy::ADTType *parent;
    TyTy::VariantDef *variant;
  };
  struct ImplItem {
    ast::AssociatedItem *implItem;
    ast::Implementation *parent;
  };
  struct TraitItem {
    TyTy::TraitReference *traitRef;
    const TyTy::TraitItemReference *itemRef;
    ast::Implementation *impl;
  };

  PathProbeCandidate(CandidateKind kind, const TyTy::BaseType *type,
                     Location loc, TraitItem trait)
      : kind(kind), type(type), loc(loc), candidate(trait) {}

  PathProbeCandidate(CandidateKind kind, const TyTy::BaseType *type,
                     Location loc, EnumItem enu)
      : kind(kind), type(type), loc(loc), candidate(enu) {}

  PathProbeCandidate(CandidateKind kind, const TyTy::BaseType *type,
                     Location loc, ImplItem impl)
      : kind(kind), type(type), loc(loc), candidate(impl) {}

  bool isEnumCandidate() const {
    return std::holds_alternative<EnumItem>(candidate);
  }
  bool isImplCandidate() const {
    return std::holds_alternative<ImplItem>(candidate);
  }
  CandidateKind getKind() const { return kind; }
  const TyTy::BaseType *getType() const { return type; }

  basic::NodeId getImplNodeId() const {
    return std::get<ImplItem>(candidate).implItem->getNodeId();
  }
  basic::NodeId getTraitNodeId() const {
    return std::get<TraitItem>(candidate).itemRef->getNodeId();
  }

  TyTy::VariantDef *getEnumVariant() const {
    return std::get<EnumItem>(candidate).variant;
  }

  ast::Implementation *getTraitImpl() const {
    return std::get<TraitItem>(candidate).impl;
  }
  ast::Implementation *getImplParent() const {
    return std::get<ImplItem>(candidate).parent;
  }

  bool operator<(const PathProbeCandidate &other) const {
    if (type < other.type)
      return true;
    return candidate.index() < other.candidate.index();
  }

private:
  CandidateKind kind;
  const TyTy::BaseType *type;
  Location loc;

  std::variant<EnumItem, ImplItem, TraitItem> candidate;
};

class MethodCandidate {
  PathProbeCandidate candidate;
  std::vector<sema::Adjustment> adjustments;

public:
  std::vector<sema::Adjustment> getAdjustments() const { return adjustments; }
  PathProbeCandidate getCandidate() const { return candidate; }
};

class PathProbeType {
public:
  PathProbeType(TyTy::BaseType *receiver, adt::Identifier &query,
                NodeId specifiedTraitId, TypeResolver *resolver)
      : receiver(receiver), query(query), specifiedTraitId(specifiedTraitId),
        resolver(resolver) {
    context = rust_compiler::session::session->getTypeContext();
  }

  static std::set<PathProbeCandidate>
  probeTypePath(TyTy::BaseType *receiver, adt::Identifier segment,
                bool probeImpls, bool probeBounds, bool ignoreTraitItems,
                TypeResolver *resolver,
                NodeId specifiedTraitId = UNKNOWN_NODEID);

protected:
  bool isReceiverGeneric();
  void processImplItemsForCandidates();
  void processEnumItemForCandidates(TyTy::ADTType *);
  void processPredicateForCandidates(const TyTy::TypeBoundPredicate &predicate,
                                     bool ignoreMandatoryTraitItems);

  void processImplItemCandidate(NodeId id, ast::Implementation *item,
                                ast::AssociatedItem *impl);

  TyTy::BaseType *receiver;
  adt::Identifier query;
  std::set<PathProbeCandidate> candidates;
  ast::AssociatedItem *currentImpl;
  NodeId specifiedTraitId;
  tyctx::TyCtx *context;
  TypeResolver *resolver;
};

} // namespace rust_compiler::sema::type_checking
