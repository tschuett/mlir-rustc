#pragma once

#include "AST/Patterns/SlicePatternItems.h"
#include "AST/Patterns/PatternWithoutRange.h"

#include <optional>

namespace rust_compiler::ast::patterns {

class SlicePattern : public PatternWithoutRange {
  std::optional<SlicePatternItems> items;

public:
  SlicePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::SlicePattern) {}

  void setPatternItems(const SlicePatternItems &item) { items = item; }
};

} // namespace rust_compiler::ast::patterns
