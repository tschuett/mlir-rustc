#pragma once

#include "AST/AST.h"
#include "AST/MacroRules.h"

namespace rust_compiler::ast {

enum class MacroRulesDefKind { Paren, Brace, Square };

class MacroRulesDef : public Node {
  MacroRulesDefKind kind;
  MacroRules rules;

public:
  MacroRulesDef(Location loc) : Node(loc), rules(loc) {}

  void setKind(MacroRulesDefKind k) { kind = k; }
  void setRules(const MacroRules &r) { rules = r; }
};

} // namespace rust_compiler::ast
