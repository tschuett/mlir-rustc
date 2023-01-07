#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

enum class LiteralExpressionKind {}

class LiteralExpression : public ExpressionWithoutBlock {
public:
  LiteralExpression(Location loc) : ExpressionWithoutBlock(loc) {}
};

} // namespace rust_compiler::ast
