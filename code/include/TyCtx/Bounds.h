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
                         const TraitItemReference *traitItemRef);

  TraitItemReference *getRawItem() const;

  bool isError() const { return parent == nullptr or traitItemRef == nullptr; }

private:
  const TypeBoundPredicate *parent;
  const TraitItemReference *traitItemRef;
};

} // namespace rust_compiler::tyctx::TyTy
