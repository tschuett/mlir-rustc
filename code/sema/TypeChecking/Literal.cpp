#include "AST/LiteralExpression.h"
#include "Basic/Mutability.h"
#include "TyCtx/TyTy.h"
#include "TypeChecking.h"

using namespace rust_compiler::ast;
using namespace rust_compiler::tyctx;
using namespace rust_compiler::tyctx::TyTy;

namespace rust_compiler::sema::type_checking {

TyTy::BaseType *
TypeResolver::checkLiteral(std::shared_ptr<ast::LiteralExpression> lit) {
  switch (lit->getLiteralKind()) {
  case ast::LiteralExpressionKind::CharLiteral: {
    TyTy::BaseType *builtin = tcx->lookupBuiltin("char");
    return builtin;
  }
  case ast::LiteralExpressionKind::StringLiteral: {
    TyTy::BaseType *builtin = tcx->lookupBuiltin("str");
    return new ReferenceType(
        lit->getNodeId(),
        TyTy::TypeVariable(builtin->getReference()), basic::Mutability::Imm);
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
    /// need size and signess hints and integral
    return new TyTy::InferType(lit->getNodeId(), TyTy::InferKind::Integral,
                               TypeHint::unknown(), lit->getLocation());
  }
  case ast::LiteralExpressionKind::FloatLiteral: {
    assert(false && "to be implemented");
  }
  case ast::LiteralExpressionKind::True:
  case ast::LiteralExpressionKind::False:
    TyTy::BaseType *builtin = tcx->lookupBuiltin("bool");
    return builtin;
  }
  assert(false);
}

} // namespace rust_compiler::sema::type_checking
