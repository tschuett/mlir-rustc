#pragma once

#include "AST/AST.h"
#include "AST/MacroMatch.h"

#include <vector>

namespace rust_compiler::ast {

enum class MacroMatcherKind { Paren, Brace, Square };

class MacroMatcher : public Node {
  MacroMatcherKind kind;
  std::vector<MacroMatch> matches;

public:
  MacroMatcher(Location loc) : Node(loc) {}

  void setKind(const MacroMatcherKind &k) { kind = k; }

  void addMatch(const MacroMatch &m) { matches.push_back(m); };
};

} // namespace rust_compiler::ast
