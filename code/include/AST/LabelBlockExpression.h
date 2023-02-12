#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class LabelBlockExpression final : public LoopExpression {
  std::shared_ptr<BlockExpression> body;

public:
  LabelBlockExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::LabelBlockExpression){};

  size_t getTokens() override;

  bool containsBreakExpression() override;
};

} // namespace rust_compiler::ast
