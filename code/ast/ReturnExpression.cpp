#include "AST/ReturnExpression.h"

namespace rust_compiler::ast {

size_t ReturnExpression::getTokens() {
  size_t count = 1;

  if (expr)
    count += expr->getTokens();

  if (hasTrailingSemi)
    ++count;

  return count;
}

} // namespace rust_compiler::ast
