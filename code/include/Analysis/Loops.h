#pragma once

#include <llvm/Support/Error.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace rust_compiler::analysis {

class LoopInfo {};

llvm::Expected<LoopInfo> detectLoop(mlir::func::FuncOp *f);

} // namespace rust_compiler::analysis
