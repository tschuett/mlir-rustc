#pragma once

#include "AST/DelimTokenTree.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/SimplePath.h"

namespace rust_compiler::ast::patterns {

class MacroInvocationPattern : public PatternWithoutRange {
  SimplePath simplePath;
  DelimTokenTree tree;

public:
  MacroInvocationPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::MacroInvocation),
        simplePath(loc), tree(loc) {}

  void setPath(const SimplePath &p) { simplePath = p; }
  void setTree(const DelimTokenTree &t) { tree = t; }
};

} // namespace rust_compiler::ast::patterns
