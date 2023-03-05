#pragma once

#include "AST/PathInExpression.h"
#include "AST/StructExpression.h"

#include <memory>

namespace rust_compiler::ast {

class StructExprUnit : public StructExpression {
  std::shared_ptr<Expression> path;

public:
  StructExprUnit(Location loc)
      : StructExpression(loc, StructExpressionKind::StructExprUnit) {
  }

  void setPath(std::shared_ptr<Expression> p) { path = p; }
};

} // namespace rust_compiler::ast
