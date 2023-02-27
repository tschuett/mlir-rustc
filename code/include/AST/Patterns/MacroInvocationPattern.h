#pragma once

#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/SimplePath.h"
#include "AST/DelimTokenTree.h"

namespace rust_compiler::ast::patterns {

class MacroInvocationPattern : public PatternWithoutRange {
  SimplePath simplePath;
  std::shared_ptr<DelimTokenTree> tree;

public:
  MacroInvocationPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::MacroInvocation),
        simplePath(loc) {}

  void setPath(const SimplePath &p) { simplePath = p; }
  void setTree(std::shared_ptr<DelimTokenTree> t) { tree = t; }
};

} // namespace rust_compiler::ast::patterns
