#include "Analysis/Attributer/Attributer.h"

#include "Analysis/Attributer/AAIsDead.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace rust_compiler::analysis::attributor {

void Attributor::setup() {
  module.walk([&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    }
    //    if (mlir::func::FuncOp *fun = mlir::dyn_cast<mlir::func::FuncOp>(op))
    //    {
    //      IRPosition FPos = IRPosition::forFuncOp(fun);
    //      getOrCreateAAFor<AAIsDead>(FPos);
    //      // for(auto& rs : fun.getResultTypes()) {
    //      //
    //      // }
    //    }
  });
}

} // namespace rust_compiler::analysis::attributor
