#include "AST/IfExpression.h"

#include "ModuleBuilder/ModuleBuilder.h"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

namespace rust_compiler {

mlir::Value
ModuleBuilder::emitIfExpression(std::shared_ptr<ast::IfExpression> ifExpr) {

  mlir::Value cond = emitExpression(ifExpr->getCondition());

  if (ifExpr->hasTrailing()) {
  } else {
  }

  //builder.create<mlir::cf::CondBranchOp>();

  assert(false);
}

} // namespace rust_compiler
