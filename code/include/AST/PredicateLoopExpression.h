#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class PredicateLoopExpression : public LoopExpression {
  std::shared_ptr<ast::Expression> condition;
  std::shared_ptr<ast::BlockExpression> block;
public:
  PredicateLoopExpression(Location loc)
      : LoopExpression(loc, LoopExpressionKind::PredicateLoopExpression) {}

  void setCondition(std::shared_ptr<ast::Expression>);
  void setBody(std::shared_ptr<ast::BlockExpression>);

  size_t getTokens() override;
  std::shared_ptr<ast::types::Type> getType() override;
};

} // namespace rust_compiler::ast
