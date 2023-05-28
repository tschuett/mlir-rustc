#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"
#include "AST/LoopLabel.h"
#include "AST/Patterns/Pattern.h"

#include <memory>
#include <optional>
#include <string>

namespace rust_compiler::ast {

class IteratorLoopExpression final : public LoopExpression {
  std::shared_ptr<patterns::Pattern> pattern;
  std::shared_ptr<Expression> rhs;
  std::shared_ptr<Expression> body;
  std::optional<LoopLabel> loopLabel;

public:
  IteratorLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::IteratorLoopExpression){};

  void setPattern(const std::shared_ptr<patterns::Pattern> &pat) {
    pattern = pat;
  }

  void setExpression(const std::shared_ptr<Expression> rh) { rhs = rh; }

  void setBody(const std::shared_ptr<Expression> bl) { body = bl; }

  void setLabel(LoopLabel lab) { loopLabel = lab; }
  bool hasLabel() const { return loopLabel.has_value(); }
  LoopLabel getLabel() const { return *loopLabel; }

  std::shared_ptr<patterns::Pattern> getPattern() const { return pattern; }
  std::shared_ptr<Expression> getBody() const { return body; }
  std::shared_ptr<Expression> getRHS() const { return rhs; }
};

} // namespace rust_compiler::ast
