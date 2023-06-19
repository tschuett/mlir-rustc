#pragma once

// #include "TraitReference.h"

namespace rust_compiler::tyctx::TyTy {

class TypeBoundPredicate;
class TraitItemReference;
class BaseType;

class TypeBoundPredicateItem {
public:
  TypeBoundPredicateItem(const TypeBoundPredicate *parent,
                         const TraitItemReference *traitItemRef)
      : parent(parent), traitItemRef(traitItemRef) {}

  const TraitItemReference *getRawItem() const { return traitItemRef; }

  bool isError() const { return parent == nullptr or traitItemRef == nullptr; }

  bool needsImplementation() const;

  TyTy::BaseType *getTypeForReceiver(const TyTy::BaseType *receiver);

  static TypeBoundPredicateItem error() {
    return TypeBoundPredicateItem(nullptr, nullptr);
  };

private:
  const TypeBoundPredicate *parent;
  const TraitItemReference *traitItemRef;
};

} // namespace rust_compiler::tyctx::TyTy
