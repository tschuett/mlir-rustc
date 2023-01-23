#pragma once

#include "AST/Expression.h"

namespace rust_compiler::ast {

enum class LoopExpressionKind {
  InfiniteLoopExpression,
  PredicateLoopExpression,
  PredicatePatternLoopExpression,
  IteratorLoopExpression,
  LabelBlockExpression
};

class LoopExpression : public ExpressionWithBlock {
  LoopExpressionKind kind;

public:
  LoopExpression(Location loc, LoopExpressionKind kind)
      : ExpressionWithBlock(loc, ExpressionWithBlockKind::LoopExpression),
        kind(kind){};

  LoopExpressionKind getLoopExpressionKind() const { return kind; }
};

} // namespace rust_compiler::ast
