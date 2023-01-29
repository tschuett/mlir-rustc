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

  size_t getTokens() override;

  bool containsBreakExpression() override;

  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast