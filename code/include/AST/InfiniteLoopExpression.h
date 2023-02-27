#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class InfiniteLoopExpression final : public LoopExpression {
  std::shared_ptr<Expression> body;
  std::optional<std::string> loopLabel;

public:
  InfiniteLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::InfiniteLoopExpression){};

  std::shared_ptr<Expression> getBody() const { return body; }

  void setBody(std::shared_ptr<Expression> bod) { body = bod; }
  void setLabel(std::string_view lab) { loopLabel = lab; }
};

} // namespace rust_compiler::ast
