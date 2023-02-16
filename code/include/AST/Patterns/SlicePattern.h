#pragma once

#include "AST/Patterns/PatternWithoutRange.h"

namespace rust_compiler::ast::patterns {

class SlicePattern : public PatternWithoutRange {

public:
  SlicePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::SlicePattern) {}
};

} // namespace rust_compiler::ast::patterns
