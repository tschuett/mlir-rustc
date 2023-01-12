#include "AST/BlockExpression.h"

#include <cassert>

namespace rust_compiler::ast {

std::shared_ptr<Statements> BlockExpression::getExpressions() { return stmts; }

void BlockExpression::setStatements(std::shared_ptr<Statements> _stmts) {
  stmts = _stmts;
}

size_t BlockExpression::getTokens() {
  size_t count = 0;

  count += stmts->getTokens();

  if (getHasTrailingSemi())
    ++count;

  return 1 + count + 1; // { }
}

} // namespace rust_compiler::ast
