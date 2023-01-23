#pragma once

#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TuplePatternItems.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast::patterns {

class TuplePattern : public PatternWithoutRange {
  std::vector<std::shared_ptr<TuplePatternItems>> items;

public:
  TuplePattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::TuplePattern) {}

  void add(std::shared_ptr<ast::patterns::TuplePatternItems> its);

  size_t getTokens() override;

  std::vector<std::string> getLiterals() override;
};

} // namespace rust_compiler::ast::patterns
