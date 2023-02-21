#include "Analysis/MemorySSA/MemorySSA.h"
#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace rust_compiler::analysis;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_DEADSTOREELIMINATIONPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class DeadStoreEliminationPass
    : public rust_compiler::optimizer::impl::DeadStoreEliminationPassBase<
          DeadStoreEliminationPass> {
public:
  void runOnOperation() override;
};

} // namespace

void DeadStoreEliminationPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  //MemorySSA &memorySSA = getAnalysis<MemorySSA>();
  module.walk([&](mlir::func::FuncOp f) {
  });
  //  isAsync() -> rewrite
  //  if (isa<rust_compiler::Mir::AwaitOp>(op)) {
  //  }
  //});
}

// https://reviews.llvm.org/D72700
