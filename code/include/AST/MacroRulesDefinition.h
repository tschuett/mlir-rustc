#pragma once

#include "AST/MacroItem.h"
#include "AST/MacroRulesDef.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast {

class MacroRulesDefinition : public MacroItem {
  std::string identifier;
  MacroRulesDef definition;

public:
  MacroRulesDefinition(Location loc)
      : MacroItem(loc, MacroItemKind::MacroRulesDefinition), definition(loc) {}

  void setIdentifier(std::string_view s) { identifier = s; }
  void setDefinition(const MacroRulesDef &d) { definition = d; }
};

} // namespace rust_compiler::ast
