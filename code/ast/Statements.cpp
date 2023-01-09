#include "AST/Statements.h"

namespace rust_compiler::ast {

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
