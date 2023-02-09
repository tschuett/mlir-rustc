#include "AST/LiteralExpression.h"

#include "AST/Types/PrimitiveTypes.h"
#include "Sema/Sema.h"

using namespace rust_compiler::ast;

namespace rust_compiler::sema {

void Sema::analyzeLiteralExpression(
    std::shared_ptr<ast::LiteralExpression> lit) {

  AstId astId = getAstId(lit);

  switch (lit->getLiteralKind()) {
  case LiteralExpressionKind::CharLiteral: {
    typeChecking.isKnownType(
        astId, std::make_shared<types::PrimitiveType>(
                   lit->getLocation(), types::PrimitiveTypeKind::Char));
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
    typeChecking.isIntegerLike(astId);
    break;
  }
  case LiteralExpressionKind::FloatLiteral: {
    typeChecking.isFloatLike(astId);
    break;
  }
  case LiteralExpressionKind::True: {
    typeChecking.isKnownType(
        astId, std::make_shared<types::PrimitiveType>(
                   lit->getLocation(), types::PrimitiveTypeKind::Boolean));
    break;
  }
  case LiteralExpressionKind::False: {
    typeChecking.isKnownType(
        astId, std::make_shared<types::PrimitiveType>(
                   lit->getLocation(), types::PrimitiveTypeKind::Boolean));
    break;
  }
  }
}

} // namespace rust_compiler::sema
