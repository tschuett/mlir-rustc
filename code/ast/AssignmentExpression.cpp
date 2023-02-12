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

bool AssignmentExpression::containsBreakExpression() {
  if (left->containsBreakExpression())
    return true;
  if (right->containsBreakExpression())
    return true;
  return false;
}

} // namespace rust_compiler::ast
