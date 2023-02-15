#include "AST/NegationExpression.h"

#include <cassert>

namespace rust_compiler::ast {

void NegationExpression::setRight(std::shared_ptr<Expression> ri) {
  right = ri;
}

void NegationExpression::setMinus() { minusToken = true; }

void NegationExpression::setNot() { notToken = true; }


} // namespace rust_compiler::ast
