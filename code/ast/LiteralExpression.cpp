#include "AST/LiteralExpression.h"

#include "AST/Types/PrimitiveTypes.h"

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

std::shared_ptr<ast::types::Type> LiteralExpression::getType() {
  switch (kind) {
  case LiteralExpressionKind::CharLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::StringLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::RawStringLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::ByteLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::ByteStringLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::RawByteStringLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::IntegerLiteral: {
    // FIXME
    return std::static_pointer_cast<types::Type>(
        std::make_shared<types::PrimitiveType>(
            getLocation(), types::PrimitiveTypeKind::Usize));
    assert(false);
    break;
  }
  case LiteralExpressionKind::FloatLiteral: {
    assert(false);
    break;
  }
  case LiteralExpressionKind::True: {
    return std::static_pointer_cast<types::Type>(
        std::make_shared<types::PrimitiveType>(
            getLocation(), types::PrimitiveTypeKind::Boolean));
  }
  case LiteralExpressionKind::False: {
    return std::static_pointer_cast<types::Type>(
        std::make_shared<types::PrimitiveType>(
            getLocation(), types::PrimitiveTypeKind::Boolean));
  }
  }
  assert(false);
  return nullptr;
}

} // namespace rust_compiler::ast
