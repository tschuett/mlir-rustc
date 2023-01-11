#pragma once

#include "AST/PatternNoTopAlt.h"
#include "AST/Patterns/PatternNoTopAlt.h

namespace rust_compiler::ast::patterns {

class TuplePattern : public PatternNoTopAlt {
public:
  TuplePattern(Location loc)
      : PatternNoTopAlt(loc, PatternNoTopAltKind::TuplePattern) {}
};

} // namespace rust_compiler::ast::patterns
