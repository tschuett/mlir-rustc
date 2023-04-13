#include "AST/LiteralExpression.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkLiteral(std::shared_ptr<ast::LiteralExpression> lit) {
  assert(false && "to be implemented");

  switch (lit->getLiteralKind()) {
  case ast::LiteralExpressionKind::CharLiteral: {
    TyTy::BaseType * builtin = tcx->lookupBuiltin("char");
    return builtin;
  }
  case ast::LiteralExpressionKind::StringLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::RawStringLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::ByteLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::ByteStringLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::RawByteStringLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::IntegerLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::FloatLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::True:
  case ast::LiteralExpressionKind::False:
    TyTy::BaseType * builtin = tcx->lookupBuiltin("bool");
    return builtin;
  }
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
