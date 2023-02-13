#pragma once

#include "AST/ConstantItem.h"
#include "AST/MacroInvocation.h"
#include "AST/OuterAttribute.h"
#include "AST/TypeAlias.h"
#include "AST/AST.h"
#include "AST/Function.h"

#include <optional>
#include <vector>

namespace rust_compiler::ast {

enum class AssociatedItemKind {
  MacroInvocation,
  TypeAlias,
  ConstantItem,
  Function
};

class AssociatedItem : public Node {
  std::vector<OuterAttribute> outerAtributes;
  AssociatedItemKind kind;

public:
  AssociatedItem(Location loc,
                 AssociatedItemKind kind)
      : Node(loc), kind(kind) {}

  AssociatedItemKind getKind() const;
};

class AssociatedItemMacroInvocation : public AssociatedItem {
  MacroInvocation macroInvocation;

public:
  AssociatedItemMacroInvocation(Location loc, std::optional<Visibility> vis)
      : AssociatedItem(loc, AssociatedItemKind::MacroInvocation),
        macroInvocation(loc) {}
};

class AssociatedItemTypeAlias : public AssociatedItem {
  TypeAlias typeAlias;

public:
  AssociatedItemTypeAlias(Location loc, std::optional<Visibility> vis)
      : AssociatedItem(loc, AssociatedItemKind::TypeAlias),
        typeAlias(loc, vis) {}
};

class AssociatedItemConstantItem : public AssociatedItem {
  ConstantItem constantItem;

public:
  AssociatedItemConstantItem(Location loc, std::optional<Visibility> vis)
      : AssociatedItem(loc, AssociatedItemKind::ConstantItem),
        constantItem(loc, vis) {}
};

class AssociatedItemFunction : public AssociatedItem {
  std::shared_ptr<Function> function;

public:
  AssociatedItemFunction(Location loc)
      : AssociatedItem(loc, AssociatedItemKind::Function) {}

  std::shared_ptr<Function> getFunction() const;
};

} // namespace rust_compiler::ast
