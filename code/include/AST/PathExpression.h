#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

class PathExpression : public ExpressionWithoutBlock {
public:
  PathExpression(rust_compiler::Location loc)
      : ExpressionWithoutBlock(loc) {}

  //  size_t getTokens() override;
};

} // namespace rust_compiler::ast
