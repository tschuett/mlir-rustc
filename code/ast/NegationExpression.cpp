#include "AST/NegationExpression.h"

#include <cassert>

namespace rust_compiler::ast {

size_t NegationExpression::getTokens() {

  assert(false);
  return 0;
}

void NegationExpression::setRight(std::shared_ptr<Expression> ri) {
  right = ri;
}

void NegationExpression::setMinus() { minusToken = true; }

void NegationExpression::setNot() { notToken = true; }

} // namespace rust_compiler::ast
