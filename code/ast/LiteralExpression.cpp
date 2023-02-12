#include "AST/LiteralExpression.h"


#include <memory>

using namespace rust_compiler::ast::types;

namespace rust_compiler::ast {

bool LiteralExpression::containsBreakExpression() { return false; }

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


} // namespace rust_compiler::ast
