#include "AST/ReturnExpression.h"

namespace rust_compiler::ast {

size_t ReturnExpression::getTokens() {
  size_t count = 1;

  if (expr)
    count += expr->getTokens();

  if (getHasTrailingSemi())
    ++count;

  return count;
}

std::shared_ptr<ast::Expression> ReturnExpression::getExpression() {
  return expr;
}

} // namespace rust_compiler::ast
