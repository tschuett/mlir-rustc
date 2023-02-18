#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"

namespace rust_compiler::ast::patterns {

class RangePattern : public PatternNoTopAlt {

public:
  RangePattern(Location loc)
      : PatternNoTopAlt(loc, PatternNoTopAltKind::RangePattern) {}
};

} // namespace rust_compiler::ast::patterns
