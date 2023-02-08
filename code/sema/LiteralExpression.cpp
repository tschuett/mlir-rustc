#include "AST/LiteralExpression.h"

#include "Sema/TypeChecking.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void TypeChecking::checkLiteralExpression(
    std::shared_ptr<ast::LiteralExpression> lit) {

  switch (lit->getLiteralKind()) {
  case LiteralExpressionKind::CharLiteral: {
    break;
  }
  case LiteralExpressionKind::StringLiteral: {
    break;
  }
  case LiteralExpressionKind::RawStringLiteral: {
    break;
  }
  case LiteralExpressionKind::RawByteStringLiteral: {
    break;
  }
  case LiteralExpressionKind::IntegerLiteral: {
    break;
  }
  case LiteralExpressionKind::FloatLiteral: {
    break;
  }
  case LiteralExpressionKind::True: {
    break;
  }
  case LiteralExpressionKind::False: {
    break;
  }
  }
}

} // namespace rust_compiler::sema
