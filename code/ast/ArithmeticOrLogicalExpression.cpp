#include "AST/ArithmeticOrLogicalExpression.h"

namespace rust_compiler::ast {

bool ArithmeticOrLogicalExpression::containsBreakExpression() { return false; }

std::shared_ptr<ast::types::Type> ArithmeticOrLogicalExpression::getType() {
  assert(false);
  return nullptr;
}

} // namespace rust_compiler::ast
