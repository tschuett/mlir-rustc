#include "AST/ComparisonExpression.h"

namespace rust_compiler::ast {

size_t ComparisonExpression::getTokens() {
  return left->getTokens() + 1 + right->getTokens();
}

std::shared_ptr<ast::types::Type> ComparisonExpression::getType() {
  assert(false);
}

} // namespace rust_compiler::ast
