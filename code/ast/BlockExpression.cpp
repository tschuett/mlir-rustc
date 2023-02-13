#include "AST/BlockExpression.h"

#include <cassert>

namespace rust_compiler::ast {

std::shared_ptr<Statements> BlockExpression::getExpressions() { return stmts; }

void BlockExpression::setStatements(std::shared_ptr<Statements> _stmts) {
  stmts = _stmts;
}

bool BlockExpression::containsBreakExpression() {
  return stmts->containsBreakExpression();
}


} // namespace rust_compiler::ast
