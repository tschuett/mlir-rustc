#include "Analysis/Loops.h"

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlow.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>

using namespace mlir;
using namespace llvm;

namespace rust_compiler::analysis {

bool isStronglyConnectComponent(const SmallPtrSetImpl<Block *> &allBlocks) {}

llvm::Expected<LoopInfo> detectLoop(mlir::func::FuncOp *f) {
  mlir::DominanceInfo domInfo;
  SmallPtrSet<Block *, 8> allBlocks;

  for (auto &block : f->getBody())
    allBlocks.insert(&block);

  for (auto &block : f->getBody()) {
    SmallPtrSet<Block *, 8> dominatedBlocks;
    for (auto &innerBlock : f->getBody()) {
      if (domInfo.dominates(&block, &innerBlock))
        dominatedBlocks.insert(&innerBlock);
    }
    if (dominatedBlocks.size() > 1) {
      // reachability check
      SmallPtrSet<Block *, 8> allLoopBlocks(dominatedBlocks);
      allLoopBlocks.insert(&block);
      if (isStronglyConnectComponent(allBlocks)) {
        //xxx;
      }
    }
  }
}

} // namespace rust_compiler::analysis
