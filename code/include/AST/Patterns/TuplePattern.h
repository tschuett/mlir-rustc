#pragma once

#include "AST/Patterns/PatternWithoutRange.h"

#include "AST/Patterns/TuplePatternItems.h"

#include <vector>
#include <memory>

namespace rust_compiler::ast::patterns {

class TuplePattern : public PatternWithoutRange {
  std::vector<std::shared_ptr<TuplePatternItems>> items;
public:
  TuplePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::TuplePattern) {}
};

} // namespace rust_compiler::ast::patterns
