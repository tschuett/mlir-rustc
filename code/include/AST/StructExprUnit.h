#pragma once

#include "AST/PathInExpression.h"
#include "AST/StructExpression.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class StructExprUnit : public StructExpression {
  PathInExpression path;

public:
  StructExprUnit(Location loc)
      : StructExpression(loc, StructExpressionKind::StructExprUnit) {}
};

} // namespace rust_compiler::ast
