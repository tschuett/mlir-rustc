#include "AST/Expression.h"

#include "Basic/Ids.h"
#include "CrateBuilder/CrateBuilder.h"
// #include <mlir/Dialect/Arith/IR/Arith.h>

#include "AST/PathExpression.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>

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

void CrateBuilder::emitReturnExpression(ast::ReturnExpression *expr) {
  if (expr->hasTailExpression()) {
    mlir::Value result = emitExpression(expr->getExpression().get());
    builder.create<mlir::func::ReturnOp>(getLocation(expr->getLocation()),
                                         result);
  }
  builder.create<mlir::func::ReturnOp>(getLocation(expr->getLocation()));

  // return builder.create<mlir::arith::ConstantIntOp
}

mlir::Value CrateBuilder::emitPathExpression(ast::PathExpression *expr) {
  std::optional<basic::NodeId> id = tyCtx->lookupName(expr->getNodeId());
  if (id) {
    std::optional<mlir::Value> value = symbolTable.find(*id);
    if (value)
      return *value;
  }
  assert(false);
}

} // namespace rust_compiler::crate_builder
