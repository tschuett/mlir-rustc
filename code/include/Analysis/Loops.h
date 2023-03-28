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
  // std::vector<>

public:
};

class Loop {
  llvm::SmallPtrSet<mlir::Block *, 8> loop;

  /// Blocks inside the loop that have successors outside of the
  /// loop. These are blocks inside the current loop, which branch
  /// out.
  llvm::SmallPtrSet<mlir::Block *, 8> exitingBlocks;

  /// Successor blocks of this loop. These are blocks outside of the
  /// current loop, which are branched to.
  llvm::SmallPtrSet<mlir::Block *, 8> exitBlocks;

  llvm::SmallPtrSet<mlir::Block *, 8> backEdges;

  mlir::Block *preHeader = nullptr;
  mlir::Block *header = nullptr;
  mlir::Block *latch = nullptr;

  /// Iff the loop's header has one unique predecessor outside of the
  /// loop. Note that this requierement is weaker than the preheader
  /// concept.
  mlir::Block *loopPredecessor = nullptr;

public:
  /// A single edge to the header of the loop from outside of the loop.
  mlir::Block getPreHeader();

  /// Single entry point of the loop that dominates all other blocks
  /// in the loop.
  mlir::Block *getHeader() const;

  /// Block that contains the branch back to the header.
  mlir::Block *getLatch() const;

  /// The block within the loop that has successors outside of the
  /// loop. If multiple blocks have successors, this is null.
  mlir::Block *getExitingBlock();

  /// The successor block of this loop. If the loop has multiple successors,
  /// this is null.
  mlir::Block *getExitBlock();

  bool hasCallOps() const { return callOp; }
  bool hasIndirectCallOps() const { return indirectCallOp; }
  bool hasReturnOps() const { return returnOp; }
  bool hasAllocOps() const { return allocOp; }
  bool hasAllocaOps() const { return allocaOp; }
  bool hasMemoryReads() const { return loadOp; }
  bool hasMemoryWrites() const { return storeOp; }
  size_t getNrOfBlocks() const;

  bool contains(mlir::Block *) const;

  friend LoopDetector;

private:
  void setHeader(Block *h) { header = h; }
  void setLatch(Block *l) { latch = l; }
  void setBlocks(llvm::SmallPtrSetImpl<Block *> &blocks) {
    loop = {blocks.begin(), blocks.end()};
  }
  void setBackEdges(llvm::SmallPtrSetImpl<Block *> &edges) {
    backEdges = {edges.begin(), edges.end()};
  }
  bool containsBlock(Block *b) { return loop.count(b) != 0; }

  llvm::SmallPtrSet<mlir::Block *, 8> getBlocks() const { return loop; }

  void findExitingBlocks();
  void findExitBlocks();
  void findLoopPredecessor();
  void findPreheader();

  void setCallOp() { callOp = true; }
  void setIndirectCallOp() { indirectCallOp = true; }
  void setReturnOp() { returnOp = true; }
  void setAllocOp() { allocOp = true; }
  void setAllocaOp() { allocaOp = true; }
  void setLoadOp() { loadOp = true; }
  void setStoreOp() { storeOp = true; }

  bool callOp = false;
  bool indirectCallOp = false;
  bool returnOp = false;
  bool allocOp = false;
  bool allocaOp = false;
  bool loadOp = false;
  bool storeOp = false;

  uint32_t level;
};

class Function {
  std::vector<Loop> loops;
  std::vector<LoopNest> nests;

  // loop nesting resp. relations
public:
  std::vector<LoopNest> getLoopNests();
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
