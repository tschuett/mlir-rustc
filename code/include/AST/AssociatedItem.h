#pragma once

#include "AST/ConstantItem.h"
#include "AST/MacroInvocation.h"
#include "AST/OuterAttribute.h"
#include "AST/TypeAlias.h"
#include "AST/VisItem.h"
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

class AssociatedItem : public VisItem {
  std::vector<OuterAttribute> outerAtributes;
  AssociatedItemKind kind;

public:
  AssociatedItem(Location loc,
                 AssociatedItemKind kind)
      : VisItem(loc, VisItemKind::AssociatedItem), kind(kind) {}

  AssociatedItemKind getKind() const;
};

class AssociatedItemMacroInvocation : public AssociatedItem {
  MacroInvocation macroInvocation;

public:
  AssociatedItemMacroInvocation(Location loc)
      : AssociatedItem(loc, AssociatedItemKind::MacroInvocation),
        macroInvocation(loc) {}
};

class AssociatedItemTypeAlias : public AssociatedItem {
  TypeAlias typeAlias;

public:
  AssociatedItemTypeAlias(Location loc)
      : AssociatedItem(loc, AssociatedItemKind::TypeAlias),
        typeAlias(loc) {}
};

class AssociatedItemConstantItem : public AssociatedItem {
  ConstantItem constantItem;

public:
  AssociatedItemConstantItem(Location loc)
      : AssociatedItem(loc, AssociatedItemKind::ConstantItem),
        constantItem(loc) {}
};

class AssociatedItemFunction : public AssociatedItem {
  std::shared_ptr<Function> function;

public:
  AssociatedItemFunction(Location loc)
      : AssociatedItem(loc, AssociatedItemKind::Function) {}

  std::shared_ptr<Function> getFunction() const;
};

} // namespace rust_compiler::ast
