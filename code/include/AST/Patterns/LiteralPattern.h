#pragma once

#include "AST/AST.h"
#include "AST/PatternWithoutRange.h"

namespace rust_compiler::ast::patterns {

class LiteralPattern : public PatternWithoutRange {
  bool isFalse = false;
  bool isTrue = false;

public:
  LiteralPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::LiteralPattern) {}

  void setTrue() { isTrue = true; }
  void setFalse() { isFalse = true; }
};

} // namespace rust_compiler::ast::patterns
