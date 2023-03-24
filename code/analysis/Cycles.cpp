#include "Analysis/Cycles.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/IR/Block.h>
#include <llvm/ADT/DenseMap.h>

namespace rust_compiler::analysis {

void CycleInfo::depthFirstSearch(mlir::Block *entryBlock) {
  llvm::SmallVector<uint32_t, 8> dfsTreeStack;
  llvm::SmallVector<mlir::Block *, 8> traverseStack;
  unsigned counter = 0;

  traverseStack.emplace_back(entryBlock);

  do {
    mlir::Block *block = traverseStack.back();

    if (!(blockDFSInfo.count(block) == 1)) {
      dfsTreeStack.emplace_back(traverseStack.size());
      llvm::append_range(traverseStack, block->getSuccessors());

      blockDFSInfo.try_emplace(block, ++counter);
      blockPreorder.push_back(block);
    } else {
      if (dfsTreeStack.back() == traverseStack.size()) {
        blockDFSInfo.find(block)->second.setEnd(counter);
        dfsTreeStack.pop_back();
      } else {
        // already done
      }
      traverseStack.pop_back();
    }
  } while (!traverseStack.empty());
}

void CycleInfo::analyze(mlir::func::FuncOp *f) {
  fun = f;

  // first step: depth first search
  depthFirstSearch(&f->getRegion().front());

  llvm::SmallVector<mlir::Block *, 8> workList;

  for (mlir::Block *headerCandidate : llvm::reverse(blockPreorder)) {
    const DFSInfo candidateInfo = blockDFSInfo.lookup(headerCandidate);

    for (mlir::Block *pred : headerCandidate->getPredecessors()) {
      const DFSInfo predDFSInfo = blockDFSInfo.lookup(headerCandidate);
      if (candidateInfo.isAncestorOf(predDFSInfo))
        workList.push_back(pred);
    }

    if (workList.empty())
      continue;

    // Found a cycle with the candidate at its header.
    std::unique_ptr<Cycle> newCycle = std::make_unique<Cycle>();
    newCycle->appendEntry(headerCandidate);
    newCycle->appendBlock(headerCandidate);
    blockMap.try_emplace(headerCandidate, newCycle.get());

    auto processPredecessors = [&](mlir::Block *block) {
      bool isEntry = false;
      for (mlir::Block *pred : block->getPredecessors()) {
        const DFSInfo predDFSInfo = blockDFSInfo.lookup(pred);
        if (candidateInfo.isAncestorOf(predDFSInfo)) {
          workList.push_back(pred);
        } else {
          isEntry = true;
        }

        if (isEntry) {
          newCycle->appendEntry(block);
        } else {
          // append as child
        }
      }
    };

    do {
      mlir::Block *block = workList.pop_back_val();
      if (block == headerCandidate)
        continue;

      if (auto *blockParent = getTopLevelParentCycle(block)) {
        if (blockParent != newCycle.get()) {
          // make blockParent the child of newCycle
          moveToNewParent(newCycle.get(), blockParent);
          newCycle->appendCyclesBlocks(blockParent);
          for (mlir::Block *childEntry : blockParent->getEntries())
            processPredecessors(childEntry);
        } else {
          // known child cycle
        }
      } else {
        blockMap.try_emplace(block, newCycle.get());
        newCycle->appendBlock(block);
        processPredecessors(block);
      }
    } while (!workList.empty());
    topLevelCycles.push_back(std::move(newCycle));
  }

  // fix top-level cycle links and compute cycle depths.
  for (auto *tlc: getTopLevelCycles()) {
    tlc->setParentCycle(nullptr);
    updateDepth(tlc);
  }
}

} // namespace rust_compiler::analysis
