#pragma once

#include "AST/Patterns/PatternWithoutRange.h"

namespace rust_compiler::ast::patterns {

class WildcardPattern : public PatternWithoutRange {

public:
  WildcardPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::WildcardPattern) {}
};

} // namespace rust_compiler::ast::patterns
