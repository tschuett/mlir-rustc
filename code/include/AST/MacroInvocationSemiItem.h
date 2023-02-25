#pragma once

#include "AST/MacroItem.h"
#include "AST/Statement.h"

#include <mlir/IR/Location.h>

namespace rust_compiler::ast {

class MacroInvocationSemiItem : public MacroItem {
  SimplePath simplePath;
  DelimTokenTree tree;

public:
  MacroInvocationSemiItem(Location loc)
    : MacroItem(loc, MacroItemKind::MacroInvocationSemi), simplePath(loc) {}

  void setPath(const SimplePath &s) { simplePath = s; }
  void setTree(const DelimTokenTree &t) { tree = t; }
};

} // namespace rust_compiler::ast
