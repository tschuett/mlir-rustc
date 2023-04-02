#pragma once

#include "AST/MacroItem.h"
#include "AST/MacroRulesDef.h"
#include "Lexer/Identifier.h"
#include "Lexer/Token.h"

#include <string>
#include <string_view>

namespace rust_compiler::ast {

using namespace rust_compiler::lexer;

class MacroRulesDefinition : public MacroItem {
  Identifier identifier;
  MacroRulesDef definition;

public:
  MacroRulesDefinition(Location loc)
      : MacroItem(loc, MacroItemKind::MacroRulesDefinition), definition(loc) {}

  void setIdentifier(const Identifier &s) { identifier = s; }
  void setDefinition(const MacroRulesDef &d) { definition = d; }
};

} // namespace rust_compiler::ast
