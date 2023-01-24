#include "ModuleBuilder/ModuleBuilder.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace mlir;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitPredicateLoopExpression(
    std::shared_ptr<ast::PredicateLoopExpression> loop) {

  OpBuilder::InsertPoint saveIP = builder.saveInsertionPoint();

  mlir::Block *entryBlock = builder.createBlock(saveIP.getBlock());

  mlir::Value cond = emitExpression(loop->getCondition());
  OpBuilder::InsertPoint condBranchIP = builder.saveInsertionPoint();

  mlir::Block *bodyBlock = builder.createBlock(entryBlock);
  emitBlockExpression(loop->getBody());
  builder.create<cf::BranchOp>(getLocation(loop->getBody()->getLocation()),
                               entryBlock);
  builder.restoreInsertionPoint(condBranchIP);
//  builder.create<cf::CondBranchOp>(
//      getLocation(loop->getCondition()->getLocation()), cond, bodyBlock, saveIP);
  builder.restoreInsertionPoint(saveIP);

  assert(false);
}

} // namespace rust_compiler
