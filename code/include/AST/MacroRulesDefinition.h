#pragma once

#include "AST/MacroItem.h"
#include "AST/Statement.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class MacroRulesDefinition : public MacroItem {

public:
  MacroRulesDefinition(Location loc)
      : MacroItem(loc, MacroItemKind::MacroRulesDefinition) {}
};

} // namespace rust_compiler::ast
