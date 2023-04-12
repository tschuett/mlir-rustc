#include "AST/Expression.h"

#include "AST/PathExpression.h"
#include "Basic/Ids.h"
#include "CrateBuilder/CrateBuilder.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace rust_compiler::crate_builder {

std::optional<mlir::Value> CrateBuilder::emitExpression(ast::Expression *expr) {
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
    std::optional<mlir::Value> result =
        emitExpression(expr->getExpression().get());
    if (result) {
      builder.create<mlir::func::ReturnOp>(getLocation(expr->getLocation()),
                                           *result);
      return;
    }
    llvm::errs() << "emitExpression in emitReturnExpression failed"
                 << "\n";
    return;
  }
  builder.create<mlir::func::ReturnOp>(getLocation(expr->getLocation()));
}

mlir::Value CrateBuilder::emitPathExpression(ast::PathExpression *expr) {
  std::optional<basic::NodeId> id = tyCtx->lookupName(expr->getNodeId());
  if (id) {
    auto it = symbolTable.begin(*id);
    if (it != symbolTable.end()) {
      return *it;
    }
  }
  assert(false);
}

} // namespace rust_compiler::crate_builder
