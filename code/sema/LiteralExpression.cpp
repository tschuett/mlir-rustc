#include "AST/LiteralExpression.h"

#include "Sema/Sema.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::basic;

namespace rust_compiler::sema {

void Sema::analyzeLiteralExpression(
    std::shared_ptr<ast::LiteralExpression> lit) {

  //  NodeId nodeId = getNodeId(lit);

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
  case LiteralExpressionKind::ByteLiteral: {
    break;
  }
  case LiteralExpressionKind::ByteStringLiteral: {
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
