#include "AST/Statements.h"

#include <memory>

namespace rust_compiler::ast {

bool Statements::containsBreakExpression() {
  for (auto &stmt : stmts) {
    if (stmt->containsBreakExpression())
      return true;
  }

  if (trailing)
    return trailing->containsBreakExpression();
}

size_t Statements::getTokens() {
  size_t count = 0;

  if (onlySemi)
    return 1;

  for (auto &stmt : stmts)
    count += stmt->getTokens();

  if (trailing)
    count += (*trailing).getTokens();

  return count;
}


} // namespace rust_compiler::ast
