#include "AST/BlockExpression.h"

namespace rust_compiler::ast {

std::span<std::shared_ptr<Statement>> BlockExpression::getExpressions() {
  return stmts;
}

} // namespace rust_compiler::ast
