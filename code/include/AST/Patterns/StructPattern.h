#pragma once

#include "AST/AST.h"
#include "AST/PathExpression.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/Patterns/StructPatternElements.h"

#include <optional>

namespace rust_compiler::ast::patterns {

class StructPattern : public PatternWithoutRange {
  std::shared_ptr<Expression> path;
  std::optional<StructPatternElements> elements;

public:
  StructPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::StructPattern) {}

  void setPath(std::shared_ptr<Expression> p) { path = p; }

  void setElements(StructPatternElements &el) { elements = el; }
};

} // namespace rust_compiler::ast::patterns
