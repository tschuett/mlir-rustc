#pragma once

#include "AST/Item.h"
#include "Location.h"

#include <optional>

namespace rust_compiler::ast {

enum class MacroItemKind {
  MacroInvocationSemi,
  MacroInvocation,
  MacroRulesDefinition
};

class MacroItem : public Item {
  MacroItemKind kind;

public:
  MacroItem(Location loc, MacroItemKind kind)
      : Item(loc, ItemKind::MacroItem), kind(kind) {}
};

} // namespace rust_compiler::ast
