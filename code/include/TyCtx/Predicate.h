#pragma once

#include "Basic/Ids.h"
#include "TyCtx/Substitutions.h"
#include "TyCtx/TraitReference.h"

namespace rust_compiler::tyctx::TyTy {

class TypeBoundPredicate : public SubstitutionRef {
public:
  // hack for std::map
  // TypeBoundPredicate() = default;
  TypeBoundPredicate(TraitReference &ref, Location loc);
  TypeBoundPredicate(basic::NodeId id,
                     std::vector<SubstitutionParamMapping> substitutions,
                     Location loc);
  virtual ~TypeBoundPredicate() = default;

  bool isError() const { return errorFlag; }

  basic::NodeId getId() const { return id; }

  lexer::Identifier getIdentifier() const { return get()->getIdentifier(); }

  static TypeBoundPredicate error() {
    TypeBoundPredicate p = TypeBoundPredicate(basic::UNKNOWN_NODEID, {},
                                              Location::getEmptyLocation());
    p.errorFlag = true;
    return p;
  }

  TraitReference *get() const;

  bool requiresGenericArgs() const;
  // FIXME resolver
  void applyGenericArgs(ast::GenericArgs *, bool hasAssociatedSelf,
                        sema::type_checking::TypeResolver *);

  BaseType *
  handleSubstitions(SubstitutionArgumentMappings &mappings) override final {
    return nullptr;
  }

  TypeBoundPredicateItem
  lookupAssociatedItem(const lexer::Identifier &search) const;

  size_t getNumberOfAssociatedBindings() const final;

  std::string toString() const;


private:
  basic::NodeId id;
  Location loc;
  bool errorFlag = false;
};

class TypeBoundsMappings {
public:
  std::vector<TypeBoundPredicate> &getSpecifiedBounds() {
    return specifiedBounds;
  }

  const std::vector<TypeBoundPredicate> &getSpecifiedBounds() const {
    return specifiedBounds;
  }

  std::string rawBoundsToString() const;

  TypeBoundPredicate lookupPredicate(basic::NodeId);
  
protected:
  TypeBoundsMappings(std::vector<TypeBoundPredicate> specifiedBounds)
      : specifiedBounds(specifiedBounds) {}

  void addBound(const TypeBoundPredicate &predicate);

private:
  std::vector<TypeBoundPredicate> specifiedBounds;
};

} // namespace rust_compiler::tyctx::TyTy
