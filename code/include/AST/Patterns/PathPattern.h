#pragma once

#include "AST/AST.h"
#include "AST/Patterns/PatternWithoutRange.h"
#include "AST/PathExpression.h"

namespace rust_compiler::ast::patterns {

class PathPattern : public PatternWithoutRange {
  std::shared_ptr<ast::Expression> path;

public:
  PathPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::PathPattern) {}

  void setPath(std::shared_ptr<ast::Expression> e) { path = e; }
};

} // namespace rust_compiler::ast::patterns
