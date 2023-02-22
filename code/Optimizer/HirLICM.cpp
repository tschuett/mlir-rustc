#include "Hir/HirOps.h"
#include "Optimizer/Passes.h"
#include "mlir/Support/LLVM.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

using namespace rust_compiler::hir;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_HIRLICMPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class HirLICMPass
    : public rust_compiler::optimizer::impl::HirLICMPassBase<HirLICMPass> {
public:
  void runOnOperation() override;
};

} // namespace

void HirLICMPass::runOnOperation() {
  mlir::func::FuncOp f = getOperation();
  for (auto &block : f.getBody()) {
    for (auto& op: block) {
      if (const auto& infi = mlir::dyn_cast<InfiniteLoopOp>(op)) {
      } else if (const auto& whil = mlir::dyn_cast<WhileLoopOp>(op)) {
      }
    }
  }
}
