#pragma once

namespace rust_compiler::ast {

class LiteralExpression : public ExpressionWithoutBlock {
public:
  LiteralExpression(Location loc) : ExpressionWithoutBlock(loc) {}
};

} // namespace rust_compiler::ast
