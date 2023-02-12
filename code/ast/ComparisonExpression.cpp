#include "AST/ComparisonExpression.h"

namespace rust_compiler::ast {

bool ComparisonExpression::containsBreakExpression() { return false; }

size_t ComparisonExpression::getTokens() {
  return left->getTokens() + 1 + right->getTokens();
}

} // namespace rust_compiler::ast
