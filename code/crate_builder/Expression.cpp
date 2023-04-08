#include "AST/Expression.h"

#include "CrateBuilder/CrateBuilder.h"

namespace rust_compiler::crate_builder {

mlir::Value CrateBuilder::emitExpression(ast::Expression *expr) {
  switch (expr->getExpressionKind()) {
  case ast::ExpressionKind::ExpressionWithBlock: {
    return emitExpressionWithBlock(
        static_cast<ast::ExpressionWithBlock *>(expr));
    break;
  }
  case ast::ExpressionKind::ExpressionWithoutBlock: {
    return emitExpressionWithoutBlock(
        static_cast<ast::ExpressionWithoutBlock *>(expr));
    break;
  }
  }
}

mlir::Value
CrateBuilder::emitMethodCallExpression(ast::MethodCallExpression *expr) {
  assert(false);
}

mlir::Value CrateBuilder::emitReturnExpression(ast::ReturnExpression *expr) {
  assert(false);
}

} // namespace rust_compiler::crate_builder
