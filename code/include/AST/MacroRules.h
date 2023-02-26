#pragma once

#include "AST/AST.h"
#include "AST/MacroRule.h"

#include <vector>

namespace rust_compiler::ast {

class MacroRules : public Node {
  std::vector<MacroRule> rules;

public:
  MacroRules(Location loc) : Node(loc) {}

  void addRule(const MacroRule &r) { rules.push_back(r); }
};

} // namespace rust_compiler::ast



// FIXME: trailing semi?
