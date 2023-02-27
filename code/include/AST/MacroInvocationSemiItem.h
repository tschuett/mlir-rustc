#pragma once

#include "AST/MacroItem.h"
#include "AST/Statement.h"
#include "AST/TokenTree.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

enum class MacroInvocationSemiItemKind { Paren, Brace, Square };

class MacroInvocationSemiItem : public MacroItem {
  MacroInvocationSemiItemKind kind;
  SimplePath simplePath;
  std::vector<TokenTree> trees;

public:
  MacroInvocationSemiItem(Location loc)
      : MacroItem(loc, MacroItemKind::MacroInvocationSemi), simplePath(loc) {}

  void setKind(MacroInvocationSemiItemKind k) { kind = k; }
  MacroInvocationSemiItemKind getKind() const { return kind; }
  void setPath(const SimplePath &s) { simplePath = s; }
  void addTree(const TokenTree& t) { trees.push_back(t); }
};

} // namespace rust_compiler::ast
