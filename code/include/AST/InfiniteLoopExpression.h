#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class InfiniteLoopExpression final : public LoopExpression {
  std::shared_ptr<Expression> body;

public:
  InfiniteLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::InfiniteLoopExpression){};

  std::shared_ptr<Expression> getBody() const { return body; }

  void setBody(std::shared_ptr<Expression> bod) { body = bod; }
};

} // namespace rust_compiler::ast
