#pragma once

#include "AST/AST.h"
#include "AST/PatternWithoutRange.h"

namespace rust_compiler::ast {

class LiteralPattern : public PatternWithoutRange {

public:
  LiteralPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::LiteralPattern) {}

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
