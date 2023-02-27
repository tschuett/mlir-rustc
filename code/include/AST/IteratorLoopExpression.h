#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"
#include "AST/Patterns/Pattern.h"

#include <memory>
#include <optional>
#include <string>


namespace rust_compiler::ast {

class IteratorLoopExpression final : public LoopExpression {
  std::shared_ptr<patterns::Pattern> pattern;
  std::shared_ptr<Expression> rhs;
  std::shared_ptr<Expression> body;
  std::optional<std::string> loopLabel;

public:
  IteratorLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::IteratorLoopExpression){};

  void setPattern(const std::shared_ptr<patterns::Pattern> &pat) {
    pattern = pat;
  }

  void setExpression(const std::shared_ptr<Expression> rh) { rhs = rh; }

  void setBody(const std::shared_ptr<Expression> bl) { body = bl; }

  void setLabel(std::string_view lab) { loopLabel = lab; }
};

} // namespace rust_compiler::ast
