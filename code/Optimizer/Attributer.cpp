#include "Analysis/Attributer/Attributer.h"

#include "Optimizer/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_ATTRIBUTERLITE
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class AttributerPass
    : public rust_compiler::optimizer::impl::AttributerLiteBase<AttributerPass> {
public:
  AttributerPass() = default;
  AttributerPass(const AttributerPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

using namespace rust_compiler::analysis::attributer;

AttributerPass::AttributerPass(const AttributerPass &pass)
    : rust_compiler::optimizer::impl::AttributerLiteBase<AttributerPass>(pass) {}

llvm::StringRef AttributerPass::getDescription() const { return "test pass"; }

void AttributerPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
//  module.walk([&](mlir::func::FuncOp *f) {
//  });

  Attributer attr = {module};
}

std::unique_ptr<mlir::Pass> createAttributerPass() {
  return std::make_unique<AttributerPass>();
}

// https://reviews.llvm.org/D140415
