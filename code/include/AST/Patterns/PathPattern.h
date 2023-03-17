#pragma once

#include "AST/AST.h"
#include "AST/PathExpression.h"
#include "AST/Patterns/PatternWithoutRange.h"

namespace rust_compiler::ast::patterns {

class PathPattern : public PatternWithoutRange {
  std::shared_ptr<ast::Expression> path;

public:
  PathPattern(Location loc)
      : PatternWithoutRange(loc, PatternWithoutRangeKind::PathPattern) {}

  void setPath(std::shared_ptr<ast::Expression> e) { path = e; }

  std::shared_ptr<ast::Expression> getPath() const { return path; }
};

} // namespace rust_compiler::ast::patterns
