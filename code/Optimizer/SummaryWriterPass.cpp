#include "Optimizer/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_SUMMARYWRITERPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class SummaryWriterPass
    : public rust_compiler::optimizer::impl::SummaryWriterPassBase<SummaryWriterPass> {
public:
  SummaryWriterPass() = default;
  SummaryWriterPass(const SummaryWriterPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

SummaryWriterPass::SummaryWriterPass(const SummaryWriterPass &pass)
    : rust_compiler::optimizer::impl::SummaryWriterPassBase<SummaryWriterPass>(pass) {}

llvm::StringRef SummaryWriterPass::getDescription() const { return "test pass"; }

void SummaryWriterPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();
//  module.walk([&](mlir::func::FuncOp *f) {
//  });
}

std::unique_ptr<mlir::Pass> createSummaryWriterPass() {
  return std::make_unique<SummaryWriterPass>();
}
