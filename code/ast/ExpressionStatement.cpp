#include "AST/ExpressionStatement.h"

namespace rust_compiler::ast {

bool ExpressionStatement::containsBreakExpression() {
  return expr->containsBreakExpression();
}

} // namespace rust_compiler::ast
