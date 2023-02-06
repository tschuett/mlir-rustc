#include "AST/MacroItem.h"

namespace rust_compiler::ast {

class MacroRulesDefinition : public MacroItem {

public:
  MacroRulesDefinition(Location loc)
      : MacroItem(loc, MacroItemKind::MacroRulesDefinition) {}
};

} // namespace rust_compiler::ast
