#pragma once

#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/TupleStructItems.h"
#include "AST/Expression.h"

#include <memory>
#include <optional>
#include <vector>

namespace rust_compiler::ast::patterns {

class TupleStructPattern : public PatternWithoutRange {
  std::shared_ptr<ast::Expression> path;
  std::optional<TupleStructItems> items;

public:
  TupleStructPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::TupleStructPattern) {}

  void setPath(std::shared_ptr<ast::Expression> p) { path = p; }

  void setItems(const TupleStructItems &it) { items = it; }

  bool hasItems() const { return items.has_value(); }

  TupleStructItems getItems() const { return *items; }

  std::shared_ptr<ast::Expression> getPath() const { return path; }
};

} // namespace rust_compiler::ast::patterns
