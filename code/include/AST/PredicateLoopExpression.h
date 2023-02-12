#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class PredicateLoopExpression final : public LoopExpression {
  std::shared_ptr<ast::Expression> condition;
  std::shared_ptr<ast::BlockExpression> block;

public:
  PredicateLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::PredicateLoopExpression) {}

  void setCondition(std::shared_ptr<ast::Expression>);
  void setBody(std::shared_ptr<ast::BlockExpression>);

  std::shared_ptr<ast::Expression> getCondition() const;
  std::shared_ptr<ast::BlockExpression> getBody() const;

      bool containsBreakExpression() override;

  size_t getTokens() override;
};

} // namespace rust_compiler::ast
