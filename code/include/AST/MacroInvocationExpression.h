#pragma once

#include "AST/Expression.h"


namespace rust_compiler::ast {

class MacroInvocationExpression : public ExpressionWithoutBlock {

public:
  MacroInvocationExpression(Location loc)
      : ExpressionWithoutBlock(loc,
                               ExpressionWithoutBlockKind::MacroInvocation) {}
};

} // namespace rust_compiler::ast
