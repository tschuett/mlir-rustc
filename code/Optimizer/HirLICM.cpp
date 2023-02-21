#include "Hir/HirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>


using namespace rust_compiler::analysis;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_HIRLICMPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class HirLICMPass
    : public rust_compiler::optimizer::impl::HirLICMPassBase<
          HirLICMPass> {
public:
  void runOnOperation() override;
};

} // namespace

void HirLICMPass::runOnOperation() {
  // mlir::func::FuncOp f = getOperation();
}

