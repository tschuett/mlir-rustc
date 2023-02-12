#include "AST/AwaitExpression.h"

namespace rust_compiler::ast {

std::shared_ptr<Expression> AwaitExpression::getBody() const { return lhs; }

bool AwaitExpression::containsBreakExpression() {
  return lhs->containsBreakExpression();
}

size_t AwaitExpression::getTokens() { return 2 + lhs->getTokens(); }

} // namespace rust_compiler::ast
