#include "AST/Expression.h"
#include "AST/LiteralExpression.h"
#include "ModuleBuilder/ModuleBuilder.h"

using namespace rust_compiler::ast;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitExpression(std::shared_ptr<Expression> expr) {
  ExpressionKind kind = expr->getExpressionKind();

  switch (kind) {
  case ExpressionKind::ExpressionWithBlock: {
    return emitExpressionWithBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithBlock>(expr));
  }
  case ExpressionKind::ExpressionWithoutBlock: {
    return buildExpressionWithoutBlock(
        static_pointer_cast<rust_compiler::ast::ExpressionWithoutBlock>(expr));
  }
  }
}

mlir::Value ModuleBuilder::emitLiteralExpression(
    std::shared_ptr<ast::LiteralExpression> lit) {
  assert(false);

  return nullptr;
}

mlir::Value ModuleBuilder::emitReturnExpression(
    std::shared_ptr<ast::ReturnExpression> ret) {
  //mlir::Value reti = emitExpression(ret->getExpression());

  assert(false);

  return nullptr;
}

mlir::Value ModuleBuilder::emitOperatorExpression(
    std::shared_ptr<ast::OperatorExpression> opr) {
  assert(false);

  return nullptr;
}

} // namespace rust_compiler
