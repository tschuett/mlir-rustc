#include "AST/IfExpression.h"

namespace rust_compiler::ast {

void IfExpression::setCondition(std::shared_ptr<ast::Expression> _condition) {
  condition = _condition;
}

std::shared_ptr<ast::Expression> IfExpression::getCondition() const {
  return condition;
}

void IfExpression::setBlock(std::shared_ptr<ast::Expression> _block) {
  block = _block;
}

std::shared_ptr<ast::Expression> IfExpression::getBlock() const {
  return block;
}

void IfExpression::setTrailing(std::shared_ptr<ast::Expression> _trailing) {
  trailing = _trailing;
}

std::shared_ptr<ast::Expression> IfExpression::getTrailing() const {
  return trailing;
}

bool IfExpression::hasTrailing() const { return !!trailing; };

} // namespace rust_compiler::ast
