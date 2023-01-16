#include "AST/LiteralExpression.h"

namespace rust_compiler::ast {

size_t LiteralExpression::getTokens() {

  if (kind == LiteralExpressionKind::IntegerLiteral)
    return 1;

  if (kind == LiteralExpressionKind::True)
    return 1;

  if (kind == LiteralExpressionKind::False)
    return 1;

  assert(false);

  return 1;
}

std::shared_ptr<ast::types::Type> LiteralExpression::getType() {
  assert(false);
  return nullptr;
}

} // namespace rust_compiler::ast
