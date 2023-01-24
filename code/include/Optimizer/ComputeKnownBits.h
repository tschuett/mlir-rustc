#pragma once

#include <llvm/Support/KnownBits.h>
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::optimizer {

llvm::KnownBits computeKnownBits(const mlir::Value *V, unsigned Depth,
                                 const mlir::Operation *CxtI);

}
