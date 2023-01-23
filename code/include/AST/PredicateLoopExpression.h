#pragma once

#include "AST/LoopExpression.h"

namespace rust_compiler::ast {

class PredicateLoopExpression : public LoopExpression {
public:
  PredicateLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::PredicateLoopExpression) {}
};

} // namespace rust_compiler::ast
