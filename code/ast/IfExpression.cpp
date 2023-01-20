#include "AST/IfExpression.h"

namespace rust_compiler::ast {

void IfExpression::setCondition(std::shared_ptr<ast::Expression> _condition) {
  condition = _condition;
}

void IfExpression::setBlock(std::shared_ptr<ast::Expression> _block) {
  block = _block;
}

void IfExpression::setTrailing(std::shared_ptr<ast::Expression> _trailing) {
  trailing = _trailing;
}

size_t IfExpression::getTokens() {
  size_t count = 1;

  count += condition->getTokens();
  count += block->getTokens();

  if (trailing)
    count += trailing->getTokens();

  return count;
}

std::shared_ptr<ast::types::Type> IfExpression::getType() {}

} // namespace rust_compiler::ast
