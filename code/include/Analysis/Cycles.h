#pragma once

#include <cstdint>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <vector>

namespace rust_compiler::analysis {

using namespace mlir;

/// A possibly irreducible generalization of a \ref Loop.
class Cycle {

  /// the entry block(s) of the cycle. The header is the only entry if
  /// this is a loop. Is empty for the root "cycle", to avoid
  /// unnecessary memory use.
  llvm::SmallVector<mlir::Block *, 4> entries;

  /// Blocks that are contained in the cycle, including entry blocks,
  /// and including blocks that are part of a child cycle.
  llvm::SmallVector<mlir::Block *, 4> blocks;

public:
  void appendEntry(mlir::Block *block) { entries.push_back(block); }
  void appendBlock(mlir::Block *block) { blocks.push_back(block); }

  bool contains(const Cycle *c) const;
  bool contains(const mlir::Block *b) const;

  mlir::Block getHeader() const;

  /// All the successor blocks of this cycle. These are blocks are outside
  /// of the current cycle which are branched to
  std::vector<mlir::Block *> getExitBlocks() const;

  Cycle *getParentCycle() const;
  unsigned getDepth() const;

  using const_entry_iterator =
    typename llvm::SmallVectorImpl<mlir::Block *>::const_iterator;

  llvm::iterator_range<const_entry_iterator> entries() const {
    return llvm::make_range(entries.begin(), entries.end());
  }
};

/// Nesting of Reducible and Irreducible Loops
/// Paul Havlak, 1997
class CycleInfo {

  class DFSInfo {
    unsigned start = 0;
    unsigned end = 0;

  public:
    DFSInfo() = default;
    explicit DFSInfo(unsigned start) : start(start) {}

    /// Whether this node is an ancestor (or equal to) the node other
    /// in the DFS tree.
    bool isAncestorOf(const DFSInfo &other) const {
      return start <= other.start and other.end <= end;
    }

    void setEnd(unsigned e) { end = e; }
  };

public:
  void analyze(mlir::func::FuncOp *f);

  Cycle *getTopLevelParentCycle(const mlir::Block *block);

private:
  /// DFS in preorder
  void depthFirstSearch(mlir::Block *entryBlock);

  void moveToNewParent(Cycle *newParent, Cycle *child);

  void updateDepth(Cycle *subTree);

  /// current function
  mlir::func::FuncOp *fun;

  llvm::DenseMap<mlir::Block *, DFSInfo> blockDFSInfo;
  llvm::SmallVector<mlir::Block *, 8> blockPreorder;

  /// map blocks to the inner-most containing loop.
  llvm::DenseMap<mlir::Block *, Cycle *> blockMap;

  /// outermost cycles discover by any DFS
  std::vector<std::unique_ptr<Cycle>> topLevelCycles;
};

} // namespace rust_compiler::analysis
