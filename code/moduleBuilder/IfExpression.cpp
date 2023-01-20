#include "AST/IfExpression.h"

#include "Mir/MirOps.h"
#include "ModuleBuilder/ModuleBuilder.h"

namespace rust_compiler {

mlir::Value
ModuleBuilder::emitIfExpression(std::shared_ptr<ast::IfExpression> ifExpr) {

  mlir::Value cond = emitExpression(ifExpr->getCondition());

  mlir::Block *currentBlock = builder.getBlock();

  if (ifExpr->hasTrailing()) {
    mlir::Block *ifBlock = builder.createBlock(currentBlock);
    mlir::Value blockValue = emitExpression(ifExpr->getBlock());

    mlir::Block *elseBlock = builder.createBlock(currentBlock);
    mlir::Value elseValue = emitExpression(ifExpr->getTrailing());

  } else {
  }

  // builder.create<Mir::CondBranchOp>(getLocation(ifExpr->getLocation()));

  assert(false);
}

} // namespace rust_compiler
