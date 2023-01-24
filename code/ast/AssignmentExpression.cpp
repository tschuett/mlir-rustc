#include "AST/AssignmentExpression.h"

namespace rust_compiler::ast {

void AssignmentExpression::setLeft(std::shared_ptr<Expression> _left) {
  left = _left;
}

void AssignmentExpression::setRight(std::shared_ptr<Expression> _right) {
  right = _right;
}

size_t AssignmentExpression::getTokens() {
  return left->getTokens() + 1 + right->getTokens();
}

std::shared_ptr<ast::types::Type> AssignmentExpression::getType() {
  assert(false);
}

} // namespace rust_compiler::ast
