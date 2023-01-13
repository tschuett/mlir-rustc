#pragma once

#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"

namespace rust_compiler::ast {

class PathInExpression : public PathExpression {
  std::vector<PathExprSegment> segs;
  uint32_t doubleColons = 0;

public:
  PathInExpression(Location loc) : PathExpression(loc) {}

  void addSegment(PathExprSegment segment) { segs.push_back(segment); }
  void addDoubleColon() { ++doubleColons; };

  size_t getTokens() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
