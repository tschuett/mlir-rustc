#include "AST/NegationExpression.h"

#include <cassert>

namespace rust_compiler::ast {

bool NegationExpression::containsBreakExpression() { return false; }

size_t NegationExpression::getTokens() {
  size_t count = 0;

  if (minusToken)
    ++count;

  if (notToken)
    ++count;

  return count + right->getTokens();
}

void NegationExpression::setRight(std::shared_ptr<Expression> ri) {
  right = ri;
}

void NegationExpression::setMinus() { minusToken = true; }

void NegationExpression::setNot() { notToken = true; }

std::shared_ptr<ast::types::Type> NegationExpression::getType() {
  return right->getType();
}

} // namespace rust_compiler::ast
