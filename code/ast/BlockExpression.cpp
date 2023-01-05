#include "AST/BlockExpression.h"

#include <cassert>

namespace rust_compiler::ast {

std::span<std::shared_ptr<Statement>> BlockExpression::getExpressions() {
  return stmts;
}

size_t BlockExpression::getTokens() {
  assert(false);
  return 0;
}

} // namespace rust_compiler::ast
