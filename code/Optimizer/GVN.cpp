#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_GVNPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class GVNPass
    : public rust_compiler::optimizer::impl::GVNPassBase<
          GVNPass> {
public:
  void runOnOperation() override;
};

} // namespace

void GVNPass::runOnOperation() {
  //  mlir::func::FuncOp module = getOperation();
  //module.walk([&](mlir::func::FuncOp f) {
  // isAsync() -> rewrite
   // if (isa<rust_compiler::Mir::AwaitOp>(op)) {
   // }
  //});
}
