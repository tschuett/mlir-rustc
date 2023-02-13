#include "AST/BorrowExpression.h"

namespace rust_compiler::ast {

bool BorrowExpression::containsBreakExpression() { return false; }

void BorrowExpression::setExpression(std::shared_ptr<Expression> _expr) {
  expr = _expr;
}

void BorrowExpression::setMut() { isMut = false; }

std::shared_ptr<Expression> BorrowExpression::getExpression() const {
  return expr;
}

bool BorrowExpression::isMutable() const { return isMut; }

} // namespace rust_compiler::ast
