#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"
#include "AST/LoopLabel.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class InfiniteLoopExpression final : public LoopExpression {
  std::shared_ptr<Expression> body;
  std::optional<LoopLabel> loopLabel;

public:
  InfiniteLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::InfiniteLoopExpression){};

  std::shared_ptr<Expression> getBody() const { return body; }

  void setBody(std::shared_ptr<Expression> bod) { body = bod; }
  void setLabel(LoopLabel ll) { loopLabel = ll; }
  bool hasLabel() const { return loopLabel.has_value(); }
  LoopLabel getLabel() const { return *loopLabel; }
};

} // namespace rust_compiler::ast
