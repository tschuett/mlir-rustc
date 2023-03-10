#pragma once

#include "AST/Patterns/PatternWithoutRange.h"

namespace rust_compiler::ast::patterns {

class RestPattern : public PatternWithoutRange {

public:
  RestPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::RestPattern) {}

};

} // namespace rust_compiler::ast::patterns
