#pragma once

#include "AST/BlockExpression.h"
#include "AST/LoopExpression.h"

#include <memory>

namespace rust_compiler::ast {

class PredicatePatternLoopExpression final : public LoopExpression {
  std::shared_ptr<ast::BlockExpression> block;

public:
  PredicatePatternLoopExpression(Location loc)
      : LoopExpression(loc,
                       LoopExpressionKind::PredicatePatternLoopExpression) {}

  //  void setCondition(std::shared_ptr<ast::Expression>);
  //  void setBody(std::shared_ptr<ast::BlockExpression>);
  //
  //  std::shared_ptr<ast::Expression> getCondition() const;
  //  std::shared_ptr<ast::BlockExpression> getBody() const;
  //
  bool containsBreakExpression() override;
  //
  size_t getTokens() override;
};

} // namespace rust_compiler::ast
