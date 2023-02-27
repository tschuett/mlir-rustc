#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/SimplePath.h"
#include "AST/Types/TypeNoBounds.h"

namespace rust_compiler::ast::types {

class MacroInvocationType : public TypeNoBounds {
  SimplePath path;
  std::shared_ptr<DelimTokenTree> tree;

public:
  MacroInvocationType(Location loc)
      : TypeNoBounds(loc, TypeNoBoundsKind::MacroInvocation), path(loc) {}

  void setPath(const SimplePath &p) { path = p; }
  void setTree(std::shared_ptr<DelimTokenTree> t) { tree = t; }
};

} // namespace rust_compiler::ast::types
