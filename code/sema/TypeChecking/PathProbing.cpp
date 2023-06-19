#include "PathProbing.h"

#include "AST/Implementation.h"
#include "AST/TraitImpl.h"
#include "Basic/Ids.h"
#include "TyCtx/Predicate.h"
#include "TypeBoundsProbe.h"

#include <llvm/Support/ErrorHandling.h>
#include <vector>

using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

std::set<PathProbeCandidate>
PathProbeType::probeTypePath(TyTy::BaseType *receiver, Identifier segment,
                             bool probeImpls, bool probeBounds,
                             bool ignoreTraitItems, TypeResolver *resolver,
                             NodeId specifiedTraitId) {
  PathProbeType probe = {receiver, segment, specifiedTraitId, resolver};

  if (probeImpls) {
    if (receiver->getKind() == TyTy::TypeKind::ADT) {
      TyTy::ADTType *adt = static_cast<TyTy::ADTType *>(receiver);
      if (adt->isEnum())
        probe.processEnumItemForCandidates(adt);
    }

    probe.processImplItemsForCandidates();
  }

  if (!probeBounds)
    return probe.candidates;

  if (!probe.isReceiverGeneric()) {
    std::vector<std::pair<TyTy::TraitReference *, ast::TraitImpl *>>
        probedBounds = TypeChecking::TypeBoundsProbe::probe(receiver, resolver);

    for (auto &candidate : probedBounds) {
      const TyTy::TraitReference *traitRef = candidate.first;
      if (specifiedTraitId != UNKNOWN_NODEID)
        if (traitRef->getNodeId() != specifiedTraitId)
          continue;
    }
  }

  for (const TyTy::TypeBoundPredicate &predicate :
       receiver->getSpecifiedBounds()) {
    const TraitReference *traitRef = predicate.get();
    if (specifiedTraitId != UNKNOWN_NODEID) {
      if (traitRef->getNodeId() != specifiedTraitId)
        continue;
    }

    probe.processPredicateForCandidates(predicate, ignoreTraitItems);
  }
  return probe.candidates;
}

bool PathProbeType::isReceiverGeneric() {
  const TyTy::BaseType *root = receiver->getRoot();
  return root->getKind() == TypeKind::Parameter ||
         root->getKind() == TypeKind::Dynamic;
}

void PathProbeType::processImplItemsForCandidates() { assert(false); }

void PathProbeType::processEnumItemForCandidates(TyTy::ADTType *adt) {
  if (specifiedTraitId != UNKNOWN_NODEID)
    return;
  TyTy::VariantDef * v;
  if (!adt->lookupVariant(query, &v))
    return;

  PathProbeCandidate::EnumItem enumItemCandidate = {adt, v};
  PathProbeCandidate candidate = {
      CandidateKind::EnumVariant, receiver->clone(),
      context->lookupLocation(adt->getTypeReference()), enumItemCandidate};

  candidates.insert(std::move(candidate));
}

void PathProbeType::processPredicateForCandidates(
    const TyTy::TypeBoundPredicate &predicate, bool ignoreMandatoryTraitItems) {
  TraitReference *traitRef = predicate.get();
  TyTy::TypeBoundPredicateItem item = predicate.lookupAssociatedItem(query);
  if (item.isError())
    return;

  if (ignoreMandatoryTraitItems && item.needsImplementation())
    return;

  const TraitItemReference *traitItemRef = item.getRawItem();
  CandidateKind candidateKind;
  switch (traitItemRef->getTraitItemKind()) {
  case TraitItemKind::Function:
    candidateKind = CandidateKind::TraitFunc;
    break;
  case TraitItemKind::Constant:
    candidateKind = CandidateKind::TraitItemConst;
    break;
  case TraitItemKind::TypeAlias:
    candidateKind = CandidateKind::TraitTypeAlias;
    break;
  case TraitItemKind::Error: {
    llvm_unreachable("no such kind");
    break;
  }
  }

  TyTy::BaseType *traitItemType = item.getRawItem()->getType();
  if (receiver->getKind() != TypeKind::Dynamic)
    traitItemType = item.getTypeForReceiver(receiver);

  PathProbeCandidate::TraitItem traitItemCandidate = {traitRef, traitItemRef,
                                                      nullptr};
  PathProbeCandidate candidate = {candidateKind, traitItemType,
                                  traitItemRef->getLocation(),
                                  traitItemCandidate};
  candidates.insert(std::move(candidate));
}

} // namespace rust_compiler::sema::type_checking
