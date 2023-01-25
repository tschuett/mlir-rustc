#include "ModuleBuilder/ModuleBuilder.h"
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

using namespace mlir;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitPredicateLoopExpression(
    std::shared_ptr<ast::PredicateLoopExpression> loop) {

  OpBuilder::InsertPoint startIP = builder.saveInsertionPoint();

  /// create blocks and save entry points
  mlir::Block *entryBlock = builder.createBlock(startIP.getBlock());
  OpBuilder::InsertPoint entryBlockIP = builder.saveInsertionPoint();
  mlir::Block *bodyBlock = builder.createBlock(entryBlock);
  OpBuilder::InsertPoint bodyIP = builder.saveInsertionPoint();
  mlir::Block *tailBlock = builder.createBlock(bodyBlock);
  OpBuilder::InsertPoint tailIP = builder.saveInsertionPoint();

  // push break point
  breakPoints.push(tailBlock);

  // branch from start block into entry block
  builder.restoreInsertionPoint(startIP);
  builder.create<cf::BranchOp>(getLocation(loop->getLocation()), entryBlock);

  // emit condition
  builder.restoreInsertionPoint(entryBlockIP);
  mlir::Value cond = emitExpression(loop->getCondition());

  // conditional branch
  builder.create<cf::CondBranchOp>(
      getLocation(loop->getCondition()->getLocation()), cond, bodyBlock,
      tailBlock);

  // emit body
  builder.restoreInsertionPoint(bodyIP);
  emitBlockExpression(loop->getBody());
  // branch to entry block
  builder.create<cf::BranchOp>(getLocation(loop->getBody()->getLocation()),
                               entryBlock);

  // restore tail ip
  builder.restoreInsertionPoint(tailIP);

  // break point
  breakPoints.pop();

  assert(false);
}

} // namespace rust_compiler
