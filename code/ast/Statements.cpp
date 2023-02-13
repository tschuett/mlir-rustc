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


} // namespace rust_compiler::ast
