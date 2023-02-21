#include "Optimizer/SCCP.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

namespace rust_compiler::optimizer {

Solver::Solver(mlir::ModuleOp &module) {
  module.walk([&](mlir::func::FuncOp f) {
    if (not f.empty()) {
      Block *entryBlock = &f.front();

      markBlockExecutable(entryBlock);

      markAllOverdefined(entryBlock->getArguments());

      localFunctions.insert(f.getSymName());
    }
  });
}

void Solver::run() {
  while (!blockWorklist.empty() || !opWorklist.empty()) {
    while (!opWorklist.empty()) {
      Operation *op = opWorklist.pop_back_val();
      // Visit all of the live users to propagate changes to this operation.
      for (Operation *user : op->getUsers()) {
        if (isBlockExecutable(user->getBlock()))
          visitOperation(user);
      }
    }

    // Process any blocks in the block worklist.
    while (!blockWorklist.empty())
      visitBlock(blockWorklist.pop_back_val());
  }
}

void Solver::visitOperation(mlir::Operation *op) {
  if (auto fun = mlir::dyn_cast<mlir::func::CallOp>(op))
    visitCallOp(&fun);
  if (auto fun = mlir::dyn_cast<mlir::func::CallIndirectOp>(op))
    visitCallIndirectOp(&fun);

  // xxx
}

void Solver::visitBlock(mlir::Block *block) {
  // If the block is not the entry block we need to compute the lattice state
  // for the block arguments. Entry block argument lattices are computed
  // elsewhere, such as when visiting the parent operation.
  if (!block->isEntryBlock()) {
    for (int i : llvm::seq<int>(0, block->getNumArguments()))
      visitBlockArgument(block, i);
  }

  // Visit all of the operations within the block.
  for (Operation &op : *block)
    visitOperation(&op);
}

void Solver::visitCallOp(mlir::func::CallOp *op) {
  if (localFunctions.contains(op->getCallee())) {
  }
}

} // namespace rust_compiler::optimizer
