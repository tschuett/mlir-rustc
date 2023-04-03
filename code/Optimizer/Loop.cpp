#include "Analysis/Loops.h"
#include "Analysis/MemorySSA/MemorySSA.h"
#include "Analysis/MemorySSA/MemorySSAWalker.h"
#include "Analysis/ScalarEvolution.h"
#include "LoopPlan.h"
#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Analysis/AliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <optional>

using namespace rust_compiler::analysis;
using namespace rust_compiler::optimizer;

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

  [[maybe_unused]]mlir::AliasAnalysis &aliasAnalysis = getAnalysis<mlir::AliasAnalysis>();
  MemorySSA &memorySSA = getAnalysis<MemorySSA>();

  [[maybe_unused]]MemorySSAWalker *walker = memorySSA.buildMemorySSA();

  LoopDetector loopDetector;
  std::optional<Function> fun = loopDetector.analyze(&f);
  if (!fun)
    return;

  std::vector<LoopNest> nests = fun->getLoopNests();

  ScalarEvolution scev;
  scev.analyze(nests);

  LoopPlanner loopPlaner = {nests, &scev, &memorySSA};
  loopPlaner.run();
}
