#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"
#include "AST/Patterns/Pattern.h"

#include <memory>

namespace rust_compiler::ast {

class IteratorLoopExpression final : public LoopExpression {
  std::shared_ptr<patterns::Pattern> pattern;
  std::shared_ptr<Expression> rhs;
  std::shared_ptr<BlockExpression> body;

public:
  IteratorLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::IteratorLoopExpression){};
};

} // namespace rust_compiler::ast
