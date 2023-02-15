#include "AST/AssignmentExpression.h"

namespace rust_compiler::ast {

void AssignmentExpression::setLeft(std::shared_ptr<Expression> _left) {
  left = _left;
}

void AssignmentExpression::setRight(std::shared_ptr<Expression> _right) {
  right = _right;
}

} // namespace rust_compiler::ast
