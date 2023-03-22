#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/RangePatternBound.h"

#include <optional>

namespace rust_compiler::ast::patterns {

enum class RangePatternKind {
  InclusiveRangePattern,
  HalfOpenRangePattern,
  ObsoleteRangePattern
};

class RangePattern : public PatternNoTopAlt {
  RangePatternKind kind;
  std::optional<RangePatternBound> lower;
  std::optional<RangePatternBound> upper;

public:
  RangePattern(Location loc)
      : PatternNoTopAlt(loc, PatternNoTopAltKind::RangePattern) {}

  void setKind(RangePatternKind k) { kind = k; }
  void setLower(const RangePatternBound &l) { lower = l; }
  void setUpper(const RangePatternBound &u) { upper = u; }
  RangePatternKind getRangeKind() const { return kind; }
};

} // namespace rust_compiler::ast::patterns
