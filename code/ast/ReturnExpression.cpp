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

std::shared_ptr<ast::types::Type> ReturnExpression::getType() {
  
}

} // namespace rust_compiler::ast
