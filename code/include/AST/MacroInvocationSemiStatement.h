#pragma once

#include "AST/SimplePath.h"
#include "AST/Statement.h"
#include "AST/TokenTree.h"
#include "Location.h"

#include <vector>

namespace rust_compiler::ast {

enum class MacroInvocationSemiStatementKind { Paren, Square, Brace };

class MacroInvocationSemiStatement : public Statement {
  MacroInvocationSemiStatementKind kind;
  SimplePath path;
  std::vector<TokenTree> trees;

public:
  MacroInvocationSemiStatement(Location loc)
      : Statement(loc, StatementKind::MacroInvocationSemi), path(loc) {}

  MacroInvocationSemiStatementKind getKind() const { return kind; }
  void setPath(const SimplePath &sp) { path = sp; }
  void addTree(const TokenTree &_tree) { trees.push_back(_tree); }
  void setKind(MacroInvocationSemiStatementKind k) { kind = k; }
};

} // namespace rust_compiler::ast
