#pragma once

#include "AST/PathExpression.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TupleStructItems.h"

#include <memory>
#include <optional>
#include <vector>

namespace rust_compiler::ast::patterns {

class TupleStructPattern : public PatternWithoutRange {
  std::shared_ptr<PathExpression> path;
  std::optional<TupleStructItems> items;

public:
  TupleStructPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::TupleStructPattern) {}

  void setPath(std::shared_ptr<PathExpression> p) { path = p; }

  void setItems(const TupleStructItems &it) { items = it; }
};

} // namespace rust_compiler::ast::patterns
