#pragma once

#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"

namespace rust_compiler::ast {

class PathInExpression final : public PathExpression {
  std::vector<PathExprSegment> segs;
  uint32_t doubleColons = 0;
  bool leadingPathSet = false;

public:
  PathInExpression(Location loc)
      : PathExpression(loc, PathExpressionKind::PathInExpression) {}

  void setLeadingPathSep() { leadingPathSet = true; }
  void addSegment(PathExprSegment segment) { segs.push_back(segment); }
  void addDoubleColon() { ++doubleColons; };

  std::vector<PathExprSegment> getSegments() const { return segs; }

  bool isSingleSegment() const { return segs.size() == 1; }
};

} // namespace rust_compiler::ast
