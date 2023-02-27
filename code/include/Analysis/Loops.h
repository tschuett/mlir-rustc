#pragma once

#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Block.h>
#include <vector>

namespace rust_compiler::analysis {

class Loop {

  llvm::SmallPtrSet<mlir::Block *, 8> loop;
  mlir::Block *preheader;
  mlir::Block *header;
  mlir::Block *latch;

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

  bool hasCallOps();
  bool hasMemoryReads();
  bool hasMemoryWrites();
};

llvm::Expected<std::vector<Loop>> detectLoop(mlir::func::FuncOp *f);

} // namespace rust_compiler::analysis
