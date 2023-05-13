#pragma once

#include "AST/Expression.h"
#include "AST/PathInExpression.h"
#include "AST/StructExpression.h"

#include <memory>
#include <vector>

namespace rust_compiler::ast {

class StructExprTuple : public StructExpression {
  std::shared_ptr<Expression> path;
  std::vector<std::shared_ptr<Expression>> exprs;

  bool trailingComma;

public:
  StructExprTuple(Location loc)
      : StructExpression(loc, StructExpressionKind::StructExprTuple) {}

  void setPath(std::shared_ptr<Expression> p) { path = p; };
  void setTrailingComma() { trailingComma = true; }
  void addExpression(std::shared_ptr<Expression> e) { exprs.push_back(e); }

  std::vector<std::shared_ptr<Expression>> &getExpressions() { return exprs; };
};

} // namespace rust_compiler::ast
