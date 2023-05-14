#pragma once

// #include "TraitReference.h"

namespace rust_compiler::tyctx {
class TraitItemReference;
}

namespace rust_compiler::tyctx::TyTy {

class TypeBoundPredicate;
// class TraitItemReference;

class TypeBoundPredicateItem {
public:
  TypeBoundPredicateItem(const TypeBoundPredicate *parent,
                         const TraitItemReference *traitItemRef)
      : parent(parent), traitItemRef(traitItemRef) {}

  const TraitItemReference *getRawItem() const { return traitItemRef; }

  bool isError() const { return parent == nullptr or traitItemRef == nullptr; }

  static TypeBoundPredicateItem error() {
    return TypeBoundPredicateItem(nullptr, nullptr);
  };

private:
  const TypeBoundPredicate *parent;
  const TraitItemReference *traitItemRef;
};

} // namespace rust_compiler::tyctx::TyTy
