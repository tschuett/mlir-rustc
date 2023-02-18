#pragma once

#include "AST/Patterns/PatternNoTopAlt.h"
#include "AST/Patterns/PatternWithoutRange.h"

#include <memory>
#include <string>

namespace rust_compiler::ast::patterns {

class ReferencePattern : public PatternWithoutRange {
  bool And = false;
  bool AndAnd = false;
  bool mut = false;
  std::shared_ptr<ast::patterns::PatternNoTopAlt> pattern;

public:
  ReferencePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::ReferencePattern) {}

  void setPattern(std::shared_ptr<ast::patterns::PatternNoTopAlt> pat) {
    pattern = pat;
  }

  void setMut() { mut = true; }
  void setAnd() { And = true; }
  void setAndAnd() { AndAnd = true; }
};

} // namespace rust_compiler::ast::patterns
