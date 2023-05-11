#pragma once

#include "AST/AssociatedItem.h"
#include "AST/Trait.h"
#include "Basic/Ids.h"
#include "Lexer/Identifier.h"
#include "TyCtx/Substitutions.h"

#include <vector>

namespace rust_compiler::ast {
class Function;
} // namespace rust_compiler::ast

namespace rust_compiler::tyctx::TyTy {
class BaseType;
}

namespace rust_compiler::tyctx {

enum class TraitItemKind { Function, Constant, TypeAlias, Error };

/// a reference to an associated item in a trait
class TraitItemReference {
public:
  TraitItemReference(const lexer::Identifier &identifier, bool isOptional,
                     TraitItemKind kind, ast::AssociatedItem *traitItem,
                     TyTy::BaseType *self,
                     std::vector<TyTy::SubstitutionParamMapping> substitutions,
                     Location loc);

  ast::AssociatedItem *getItem() const { return item; }
  basic::NodeId getNodeId() const { return item->getNodeId(); }

  static TraitItemReference error() {
    return TraitItemReference(lexer::Identifier(""), false,
                              TraitItemKind::Error, nullptr, nullptr, {},
                              Location::getEmptyLocation());
  }

  static TraitItemReference &error_node() {
    static TraitItemReference error = TraitItemReference::error();
    return error;
  }

  std::string traitItemTypeToString() const;

  TyTy::BaseType *getType() const { return self; }

  std::string toString() const;

private:
  lexer::Identifier identifier;
  bool isOptional;
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

  static TraitReference &error_node() {
    static TraitReference traitErrorNode = TraitReference::error();
    return traitErrorNode;
  }

  basic::NodeId getNodeId() const { return trait->getNodeId(); }

  std::vector<TyTy::SubstitutionParamMapping> getTraitSubsts() const {
    return substs;
  };

  std::string toString() const;

private:
  ast::Trait *trait;
  std::vector<TraitItemReference> items;
  std::vector<const TraitReference *> superTraits;
  std::vector<TyTy::SubstitutionParamMapping> substs;
};

} // namespace rust_compiler::tyctx
