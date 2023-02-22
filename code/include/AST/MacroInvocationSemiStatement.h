#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/SimplePath.h"
#include "AST/Statement.h"
#include "Location.h"

namespace rust_compiler::ast {

class MacroInvocationSemiStatement : public Statement {
  SimplePath path;
  DelimTokenTree tree;

public:
  MacroInvocationSemiStatement(Location loc)
    : Statement(loc, StatementKind::MacroInvocationSemi), path(loc) {}

  void setPath(const SimplePath &sp) { path = sp; }
  void setTree(const DelimTokenTree &_tree) { tree = _tree; }
};

} // namespace rust_compiler::ast
