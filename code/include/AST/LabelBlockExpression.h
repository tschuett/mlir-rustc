#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace rust_compiler::ast {

class LabelBlockExpression final : public LoopExpression {
  std::shared_ptr<Expression> body;
  std::optional<std::string> loopLabel;

public:
  LabelBlockExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::LabelBlockExpression){};

  void setLabel(std::string_view l) { loopLabel = l; }
  void setBlock(std::shared_ptr<Expression> b) { body = b; }

  std::shared_ptr<Expression> getBody() const { return body; }
};

} // namespace rust_compiler::ast
