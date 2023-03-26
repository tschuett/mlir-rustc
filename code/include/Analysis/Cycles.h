#pragma once

#include <cstdint>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/SmallVector.h>
#include <memory>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/ControlFlow/IR/ControlFlowOps.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Dominance.h>
#include <vector>

namespace rust_compiler::analysis {

using namespace mlir;

/// A natural loop.
class Loop {

  /// the entry block(s) of the loop. The header is the only entry if
  /// this is a loop. Is empty for the root "cycle", to avoid
  /// unnecessary memory use.
  llvm::SmallVector<mlir::Block *, 4> entries;

  /// Blocks that are contained in the cycle, including entry blocks,
  /// and including blocks that are part of a child cycle.
  llvm::SmallVector<mlir::Block *, 4> blocks;

  mlir::Block *header = nullptr;

public:
  void appendEntry(mlir::Block *block) { entries.push_back(block); }
  void appendBlock(mlir::Block *block) { blocks.push_back(block); }
  void appendCyclesBlocks(Cycle *cycle) {
    entries.insert(blocks.end(), cycle->blocks.begin(), cycle->blocks.end());
  }

  bool contains(const mlir::Block *b) const;

  mlir::Block *getHeader() const { return header; }

  /// All the successor blocks of this cycle. These are blocks are outside
  /// of the current cycle which are branched to
  std::vector<mlir::Block *> getExitBlocks() const;

  /// All blocks inside the loop that have successors outside of the
  /// loop.
  std::vector<mlir::Block *> getExitingBlocks() const;

  using const_entry_iterator =
      typename llvm::SmallVectorImpl<mlir::Block *>::const_iterator;

  llvm::iterator_range<const_entry_iterator> getBlocks() const {
    return llvm::make_range(blocks.begin(), blocks.end());
  }

  /// A latch block contains the branch back to the header
  mlir::Block *getLatch() const;

  /// All latch blocks
  std::vector<mlir::Block *> getLatches() const;

  /// get the latch condition instruction
  mlir::arith::CmpIOp *getLatchCmpInst() const;

  /// Check if the loop has a canonical induction variable
  std::optional<BlockArgument> getCanonicalInductionVariable() const;

  /// A loop has a preheader if there is only one edge to the header
  /// from outside of the loop.
  mlir::Block *getLoopPreHeader() const;

  /// If the loop's header has exactly one unique predecessor outside
  /// of the loop. This is less strict than preheader.
  mlir::Block *getLoopPredecessor() const;

private:
  mlir::DominanceInfo dominanceInfo;
};

class LoopInfo {

public:
  void analyze(mlir::func::FuncOp *f);

private:
  /// current function
  mlir::func::FuncOp *fun;

  /// map blocks to the inner-most containing loop.
  llvm::DenseMap<mlir::Block *, Loop *> blockMap;

  /// loops discovered
  std::vector<std::unique_ptr<Loop>> loops;
};

} // namespace rust_compiler::analysis
