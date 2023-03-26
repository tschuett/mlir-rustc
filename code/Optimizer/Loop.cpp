#include "Analysis/Loops.h"
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
#define GEN_PASS_DEF_LOOPPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class LoopPass : public rust_compiler::optimizer::impl::LoopPassBase<LoopPass> {
public:
  void runOnOperation() override;
};

} // namespace

void LoopPass::runOnOperation() {
  mlir::func::FuncOp f = getOperation();

  LoopDetector loopDetector;
  loopDetector.analyze(&f);
}
