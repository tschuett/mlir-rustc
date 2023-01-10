#include "Analysis/Attributer/Attributer.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Operation.h"

namespace rust_compiler::analysis::attributer {

void Attributer::setup() {
  module.walk([&](mlir::Operation *op) {
    if (auto call = mlir::dyn_cast<mlir::func::CallOp>(op)) {
    }
    if (auto fun = mlir::dyn_cast<mlir::func::FuncOp>(op)) {
      //for(auto& rs : fun.getResultTypes()) {
      //  
      //}
    }
  });
}

} // namespace rust_compiler::analysis::attributer
