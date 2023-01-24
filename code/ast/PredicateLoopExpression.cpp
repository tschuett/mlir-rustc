#include "AST/PredicateLoopExpression.h"

namespace rust_compiler::ast {

void PredicateLoopExpression::setCondition(
    std::shared_ptr<ast::Expression> _cond) {
  condition = _cond;
}
void PredicateLoopExpression::setBody(
    std::shared_ptr<ast::BlockExpression> _body) {
  block = _body;
}

size_t PredicateLoopExpression::getTokens() {
  size_t count = 0;
  count += condition->getTokens();
  count += block->getTokens();

  return 1 + count;
}

std::shared_ptr<ast::Expression> PredicateLoopExpression::getCondition() const {
  return condition;
}

std::shared_ptr<ast::BlockExpression> PredicateLoopExpression::getBody() const {
  return block;
}

std::shared_ptr<ast::types::Type> PredicateLoopExpression::getType() {
  assert(false);
}

} // namespace rust_compiler::ast
