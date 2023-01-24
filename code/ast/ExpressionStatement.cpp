#include "AST/ExpressionStatement.h"

namespace rust_compiler::ast {

bool ExpressionStatement::containsBreakExpression() {
  return expr->containsBreakExpression();
}

size_t ExpressionStatement::getTokens() { return 1 + expr->getTokens(); }

} // namespace rust_compiler::ast
