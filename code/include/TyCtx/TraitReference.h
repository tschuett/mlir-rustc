#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Implementation.h"
#include "AST/Trait.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "TyCtx/Substitutions.h"

#include <optional>
#include <vector>

namespace rust_compiler::ast {
class Function;
} // namespace rust_compiler::ast

namespace rust_compiler::tyctx::TyTy {
class BaseType;
class TyCtx;
} // namespace rust_compiler::tyctx::TyTy

namespace rust_compiler::tyctx::TyTy {

enum class TraitItemKind { Function, Constant, TypeAlias, Error };

/// a reference to an associated item in a trait
class TraitItemReference {
public:
  TraitItemReference(const lexer::Identifier &identifier, bool isOptional,
                     TraitItemKind kind, ast::AssociatedItem *traitItem,
                     TyTy::BaseType *self,
                     std::vector<TyTy::SubstitutionParamMapping> substitutions,
                     Location loc)
      : identifier(identifier), isOptional2(isOptional), kind(kind),
        item(traitItem), self(self), substitutions(substitutions), loc(loc) {}

  ast::AssociatedItem *getItem() const { return item; }
  basic::NodeId getNodeId() const { return item->getNodeId(); }

  static TraitItemReference error() {
    return TraitItemReference(lexer::Identifier(""), false,
                              TraitItemKind::Error, nullptr, nullptr, {},
                              Location::getEmptyLocation());
  }

  static TraitItemReference &errorNode() {
    static TraitItemReference error = TraitItemReference::error();
    return error;
  }

  bool isType() const { return item->hasTypeAlias(); }
  std::string traitItemTypeToString() const;

  TyTy::BaseType *getType() const { return self; }

  std::string toString() const;

  void associatedTypeSet(TyTy::BaseType *) const;

  lexer::Identifier getIdentifier() const { return identifier; }

  TraitItemKind getTraitItemKind() const { return kind; }

  Location getLocation() const { return loc; }

  bool isOptional() const { return isOptional2; }

private:
  lexer::Identifier identifier;
  bool isOptional2;
  TraitItemKind kind;
  ast::AssociatedItem *item;
  TyTy::BaseType *self;
  std::vector<TyTy::SubstitutionParamMapping> substitutions;
  Location loc;

  void resolveFunction(ast::Function *);
};

/// a reference to a Trait
class TraitReference {

public:
  TraitReference(ast::Trait *ref, std::vector<TraitItemReference> items,
                 std::vector<const TraitReference *> superTraits,
                 std::vector<TyTy::SubstitutionParamMapping> substs)
      : trait(ref), items(items), superTraits(superTraits), substs(substs) {}

  ast::Trait *getTrait() const { return trait; }

  bool isError() const { return trait == nullptr; }

  static TraitReference error() { return TraitReference(nullptr, {}, {}, {}); }

  lexer::Identifier getIdentifier() const { return trait->getIdentifier(); }

  static TraitReference &errorNode() {
    static TraitReference traitErrorNode = TraitReference::error();
    return traitErrorNode;
  }

  basic::NodeId getNodeId() const { return trait->getNodeId(); }

  std::vector<TyTy::SubstitutionParamMapping> getTraitSubsts() const {
    return substs;
  };

  std::string toString() const;

  const std::vector<TraitItemReference> &getTraitItems() const { return items; }

  std::optional<TraitItemReference *>
  lookupTraitItem(const lexer::Identifier &);

private:
  ast::Trait *trait;
  std::vector<TraitItemReference> items;
  std::vector<const TraitReference *> superTraits;
  std::vector<TyTy::SubstitutionParamMapping> substs;
};

} // namespace rust_compiler::tyctx
