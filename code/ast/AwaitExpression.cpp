#include "AST/AwaitExpression.h"

namespace rust_compiler::ast {

std::shared_ptr<Expression> AwaitExpression::getBody() const { return lhs; }

bool AwaitExpression::containsBreakExpression() {
  return lhs->containsBreakExpression();
}

} // namespace rust_compiler::ast
