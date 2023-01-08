#include "AST/ReturnExpression.h"

namespace rust_compiler::ast {

size_t ReturnExpression::getTokens() {

  if (expr)
    return 1+ expr->getTokens();

  return 1;
}

} // namespace rust_compiler::ast
