#include "AST/BorrowExpression.h"

namespace rust_compiler::ast {

size_t BorrowExpression::getTokens() {

  if (isMut)
    return 1 + expr->getTokens();

  return expr->getTokens();
}

void BorrowExpression::setExpression(std::shared_ptr<Expression> _expr) {
  expr = _expr;
}

void BorrowExpression::setMut() { isMut = false; }

} // namespace rust_compiler::ast
