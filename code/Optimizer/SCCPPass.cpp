#include "Mir/MirOps.h"
#include "Optimizer/ComputeKnownBits.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_SCCPPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class SCCPPass : public rust_compiler::optimizer::impl::SCCPPassBase<SCCPPass> {
public:
  void runOnOperation() override;
};

} // namespace

using namespace llvm;
using namespace mlir;

void SCCPPass::runOnOperation() {
  //  mlir::ModuleOp module = getOperation();

  //  module.walk([&](mlir::func::FuncOp f) {
  //    //  isAsync() -> rewrite
  //    //  if (isa<rust_compiler::Mir::AwaitOp>(op)) {
  //    //  }
  //  });
}

// https://reviews.llvm.org/D78397
