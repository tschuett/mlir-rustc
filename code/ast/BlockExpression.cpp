#include "AST/BlockExpression.h"

#include <cassert>

namespace rust_compiler::ast {

std::shared_ptr<Statements> BlockExpression::getExpressions() {
  return stmts;
}

void BlockExpression::setStatements(std::shared_ptr<Statements> _stmts) {
  stmts = _stmts;
}

size_t BlockExpression::getTokens() {
  assert(false);
  return 0;
}

} // namespace rust_compiler::ast
