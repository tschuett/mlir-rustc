#pragma once

#include "AST/PathExprSegment.h"
#include "AST/PathExpression.h"

namespace rust_compiler::ast {

class PathInExpression : public PathExpression {
public:
  PathInExpression(Location loc) : PathExpression(loc) {}
  size_t getTokens() override;

  void addSegment(PathExprSegment segment);
};

} // namespace rust_compiler::ast
