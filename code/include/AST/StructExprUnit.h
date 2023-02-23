#pragma once

#include "AST/PathInExpression.h"
#include "AST/StructExpression.h"

#include <memory>

namespace rust_compiler::ast {

class StructExprUnit : public StructExpression {
  std::shared_ptr<PathExpression> path;

public:
  StructExprUnit(Location loc)
      : StructExpression(loc, StructExpressionKind::StructExprUnit) {
  }

  void setPath(std::shared_ptr<PathExpression> p) { path = p; }
};

} // namespace rust_compiler::ast
