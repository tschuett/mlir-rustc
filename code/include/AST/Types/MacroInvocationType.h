#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/SimplePath.h"
#include "AST/Types/TypeNoBounds.h"

namespace rust_compiler::ast::types {

class MacroInvocationType : public TypeNoBounds {
  SimplePath path;
  DelimTokenTree tree;

public:
  MacroInvocationType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::MacroInvocation), path(loc),
        tree(loc) {}

  void setPath(const SimplePath &p) { path = p; }
  void setTree(const DelimTokenTree &t) { tree = t; }
};

} // namespace rust_compiler::ast::types
