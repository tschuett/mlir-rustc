#include "TyCtx/Bounds.h"

#include "TyCtx/Substitutions.h"
#include "TyCtx/SubstitutionsMapper.h"
#include "TyCtx/TraitReference.h"
#include "TyCtx/TyTy.h"

namespace rust_compiler::tyctx::TyTy {

bool TypeBoundPredicateItem::needsImplementation() const {
  return !getRawItem()->isOptional();
}

TyTy::BaseType *
TypeBoundPredicateItem::getTypeForReceiver(const TyTy::BaseType *receiver) {
  TyTy::BaseType *traitItemType = getRawItem()->getType();
  if (parent->getSubstitutionArguments().isEmpty())
    return traitItemType;

  const TraitItemReference *tref = getRawItem();
  // FIXME: check
  if (tref->getTraitItemKind() == TraitItemKind::Function)
    return traitItemType;

  SubstitutionArgumentMappings gargs = parent->getSubstitutionArguments();
  assert(!gargs.isEmpty());

  std::vector<SubstitutionArg> adjustedMappings;
  for (size_t i = 0; i < gargs.getMappings().size(); ++i) {
    auto &mapping = gargs.getMappings()[i];

    TyTy::BaseType *argument = (i = 0) ? receiver->clone() : mapping.getType();
    SubstitutionArg arg = {mapping.getParamMapping(), argument};
    adjustedMappings.push_back(std::move(arg));
  }

  SubstitutionArgumentMappings adjusted = {
      adjustedMappings, {}, gargs.getLocation(), gargs.getSubstCb(), true};
  InternalSubstitutionsMapper mapper;
  return mapper.resolve(traitItemType, adjusted);
}

} // namespace rust_compiler::tyctx::TyTy
