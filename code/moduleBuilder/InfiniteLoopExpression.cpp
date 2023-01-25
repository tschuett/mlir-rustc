#include "ModuleBuilder/ModuleBuilder.h"

#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>

using namespace mlir;

namespace rust_compiler {

mlir::Value ModuleBuilder::emitInfiniteLoopExpression(
    std::shared_ptr<ast::InfiniteLoopExpression> infi) {

  // save the start IP
  OpBuilder::InsertPoint startIP = builder.saveInsertionPoint();

  /// create blocks and save entry points
  mlir::Block *bodyBlock = builder.createBlock(entryBlock);
  OpBuilder::InsertPoint bodyIP = builder.saveInsertionPoint();
  mlir::Block *tailBlock = builder.createBlock(bodyBlock);
  //OpBuilder::InsertPoint tailIP = builder.saveInsertionPoint();

  // push break point
  breakPoints.push(tailBlock);

  // branch from start block into entry block
  builder.restoreInsertionPoint(startIP);
  builder.create<cf::BranchOp>(getLocation(infi->getLocation()), bodyBlock);

  // restore body IP
  builder.restoreInsertionPoint(bodyIP);
  emitBlockExpression(infi->getBody());
  builder.create<cf::BranchOp>(getLocation(infi->getLocation()), bodyBlock);

  // break point
  breakPoints.pop();

  assert(false);
}

} // namespace rust_compiler


// FIXME tail block must have block arguments matching break expression
