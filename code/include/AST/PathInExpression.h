#pragma once

#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"

namespace rust_compiler::ast {

class PathInExpression : public PathExpression {
  std::vector<PathExprSegment> segs;
  uint32_t doubleColons = 0;

public:
  PathInExpression(Location loc) : PathExpression(loc) {}
  size_t getTokens() override;

  void addSegment(PathExprSegment segment) { segs.push_back(segment); }
  void addDoubleColon() { ++doubleColons; };
};

} // namespace rust_compiler::ast
