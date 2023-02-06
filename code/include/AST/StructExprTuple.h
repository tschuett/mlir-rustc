#pragma once

#include "AST/Expression.h"
#include "AST/PathInExpression.h"
#include "AST/StructExpression.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class StructExprTuple : public StructExpression {
  PathInExpression path;
  std::vector<std::shared_ptr<Expression>> exprs;

  bool trailingComma;
  bool trailingBracket;

public:
  StructExprTuple(Location loc)
      : StructExpression(loc, StructExpressionKind::StructExprTuple) {}
};

} // namespace rust_compiler::ast
