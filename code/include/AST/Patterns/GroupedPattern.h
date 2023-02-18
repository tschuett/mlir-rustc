#pragma once

#include "AST/Patterns/Pattern.h"
#include "AST/Patterns/PatternWithoutRange.h"

#include <memory>

namespace rust_compiler::ast::patterns {

class GroupedPattern : public PatternWithoutRange {
  std::shared_ptr<ast::patterns::Pattern> pattern;

public:
  GroupedPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::GroupedPattern) {}

  void setPattern(std::shared_ptr<ast::patterns::Pattern> pat) {
    pattern = pat;
  }
};

} // namespace rust_compiler::ast::patterns
