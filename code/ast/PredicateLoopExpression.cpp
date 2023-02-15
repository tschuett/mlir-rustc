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

std::shared_ptr<ast::Expression> PredicateLoopExpression::getCondition() const {
  return condition;
}

std::shared_ptr<ast::BlockExpression> PredicateLoopExpression::getBody() const {
  return block;
}

} // namespace rust_compiler::ast
