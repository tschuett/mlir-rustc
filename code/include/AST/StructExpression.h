#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

enum class StructExpressionKind {
  StructExprStruct,
  StructExprTuple,
  StructExprUnit
};

class StructExpression : public ExpressionWithoutBlock {
  StructExpressionKind kind;

public:
  StructExpression(Location loc, StructExpressionKind kind)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::StructExpression),
        kind(kind) {}

  StructExpressionKind getKind() const { return kind; };
};

} // namespace rust_compiler::ast
