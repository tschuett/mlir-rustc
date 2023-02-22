#pragma once

#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class PredicateLoopExpression final : public LoopExpression {
  std::shared_ptr<ast::Expression> condition;
  std::shared_ptr<ast::Expression> block;

public:
  PredicateLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::PredicateLoopExpression) {}

  void setCondition(std::shared_ptr<ast::Expression> cond) { condition = cond; }
  void setBody(std::shared_ptr<ast::Expression> b) { block = b; }

  std::shared_ptr<ast::Expression> getCondition() const { return condition; }
  std::shared_ptr<ast::Expression> getBody() const { return block; }
};

} // namespace rust_compiler::ast
