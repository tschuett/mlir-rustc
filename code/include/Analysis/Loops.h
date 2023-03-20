#pragma once

#include <llvm/ADT/SmallPtrSet.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <vector>

namespace rust_compiler::analysis {

using namespace mlir;

/// https://arxiv.org/pdf/1811.00632.pdf
/// https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Analysis/LoopInfoImpl.h
/// https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/Analysis/LoopInfo.h

class LoopDetector;
  class Loop;

class LoopNest {
  Loop *topLoop;
  //std::vector<>
};

class Loop {
  llvm::SmallPtrSet<mlir::Block *, 8> loop;
  llvm::SmallPtrSet<mlir::Block *, 8> exitNodes;
  llvm::SmallPtrSet<mlir::Block *, 8> backEdges;

  mlir::Block *preheader = nullptr;
  mlir::Block *header = nullptr;
  mlir::Block *latch = nullptr;

public:
  /// A single edge to the header of the loop from outside of the loop.
  mlir::Block getPreHeader();

  /// Single entry point of the loop that dominates all other blocks
  /// in the loop.
  mlir::Block *getHeader();

  /// Block that contains the branch back to the header.
  mlir::Block *getLatch();

  /// The block within the loop that has successors outside of the
  /// loop. If multiple blocks have successors, this is null.
  mlir::Block *getExitingBlock();

  /// The successor block of this loop. If the loop has multiple successors,
  /// this is null.
  mlir::Block *getExitBlock();

  bool hasCallOps() const;
  bool hasMemoryReads() const;
  bool hasMemoryWrites() const;
  size_t getNrOfBlocks() const;

  friend LoopDetector;

private:
  void setHeader(Block *h) { header = h; }
  void setLatch(Block *l) { latch = l; }
  void setPreHeader(Block *p) { preheader = p; }
  void setBlocks(llvm::SmallPtrSetImpl<Block *> &blocks) {
    loop = {blocks.begin(), blocks.end()};
  }
  void setBackEdges(llvm::SmallPtrSetImpl<Block *> &edges) {
    backEdges = {edges.begin(), edges.end()};
  }
  bool containsBlock(Block *b) { return loop.count(b) != 0; }

  llvm::SmallPtrSet<mlir::Block *, 8> getBlocks() const { return loop; }

  uint32_t level;
};

class Function {
  std::vector<Loop> loops;

  // loop nesting resp. relations
};

class LoopDetector {
public:
  std::optional<Function> analyze(mlir::func::FuncOp *f);

private:
  void detectLoopCandidates();
  void createLoop(llvm::SmallPtrSetImpl<mlir::Block *> &scc,
                  mlir::Block *header);
  void analyzeRelationShips();
  /// canonical 5 nested loops: how to detect? and precise nesting
  void analyzeLoopNests();
  void analyzeInductionVariable(Loop *l);

  bool doSetsOverlap(llvm::SmallPtrSetImpl<Block *> &first,
                     llvm::SmallPtrSetImpl<Block *> &second);

  bool doesSetContains(llvm::SmallPtrSetImpl<Block *> &first,
                       llvm::SmallPtrSetImpl<Block *> &second);

  // dominator tree
  mlir::DominanceInfo domInfo;

  mlir::func::FuncOp *f;
  Function fun;

  // candidates
  std::vector<Loop> loops;
};

} // namespace rust_compiler::analysis
