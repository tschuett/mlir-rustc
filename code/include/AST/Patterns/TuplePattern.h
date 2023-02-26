#pragma once

#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TuplePatternItems.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class TuplePattern : public PatternWithoutRange {
  TuplePatternItems items;

public:
  TuplePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::TuplePattern),
        items(loc) {}

  void setItems(const TuplePatternItems &its) { items = its; }
};

} // namespace rust_compiler::ast::patterns
