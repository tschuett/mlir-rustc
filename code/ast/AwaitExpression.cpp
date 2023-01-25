#include "AST/AwaitExpression.h"

namespace rust_compiler::ast {

std::shared_ptr<Expression> AwaitExpression::getBody() const { return lhs; }

bool AwaitExpression::containsBreakExpression() {
  return lhs->containsBreakExpression();
}

size_t AwaitExpression::getTokens() { return 2 + lhs->getTokens(); }

std::shared_ptr<ast::types::Type> AwaitExpression::getType() { assert(false); }

} // namespace rust_compiler::ast
