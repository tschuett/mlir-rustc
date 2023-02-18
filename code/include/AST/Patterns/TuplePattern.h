#pragma once

#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TuplePatternItems.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class TuplePattern : public PatternWithoutRange {
  std::vector<TuplePatternItems> items;

public:
  TuplePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::TuplePattern) {}

  void addItems(const TuplePatternItems& its) {
    items.push_back(its);
  }
};

} // namespace rust_compiler::ast::patterns
