#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class InfiniteLoopExpression final : public LoopExpression {
  std::shared_ptr<BlockExpression> body;

public:
  InfiniteLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::InfiniteLoopExpression){};

  std::shared_ptr<BlockExpression> getBody() const;

  bool containsBreakExpression() override;
};

} // namespace rust_compiler::ast
