#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_FUNCSPECIALPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class FuncSpecialPass
    : public rust_compiler::optimizer::impl::FuncSpecialPassBase<
          FuncSpecialPass> {
public:
  void runOnOperation() override;

private:
  void checkCallOps(mlir::func::FuncOp *f);
};

} // namespace

void FuncSpecialPass::checkCallOps(mlir::func::FuncOp *f) {
  for (auto &block : f->getBody()) {
    for (auto &op : block.getOperations()) {
      if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
        /// check args
      }
    }
  }
}

void FuncSpecialPass::runOnOperation() {
  //  memorySSA &MemorySSA = getAnalysis<MemorySSA>();
  mlir::ModuleOp module = getOperation();
  module.walk([&](mlir::func::FuncOp f) { checkCallOps(&f); });
}
