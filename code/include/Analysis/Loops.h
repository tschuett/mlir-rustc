#pragma once

#include "mlir/IR/Block.h"

#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <vector>

namespace rust_compiler::analysis {

class Loop {

public:
  mlir::Block getPreHeader();
  mlir::Block *getHeader();
  mlir::Block *getLatch();
  mlir::Block *getExitingBlock();
  mlir::Block *getExitBlock();

  bool hasCallOps();
  bool hasMemoryReads();
  bool hasMemoryWrites();
};

class LoopInfo {};

llvm::Expected<std::vector<Loop>> detectLoop(mlir::func::FuncOp *f);

} // namespace rust_compiler::analysis
