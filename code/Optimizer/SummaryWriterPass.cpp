#include "Mir/MirDialect.h"
#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include <llvm/Demangle/Demangle.h>
#include <mlir/IR/BuiltinOps.h>
#include <string>

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_SUMMARYWRITERPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class SummaryWriterPass
    : public rust_compiler::optimizer::impl::SummaryWriterPassBase<
          SummaryWriterPass> {
public:
  //  SummaryWriterPass() = default;
  // SummaryWriterPass(const SummaryWriterPass &pass);

  SummaryWriterPass(
      const rust_compiler::optimizer::SummaryWriterPassOptions &options)
      : SummaryWriterPassBase(options) {}

  SummaryWriterPass() : SummaryWriterPassBase() {}

  void runOnOperation() override;
};

} // namespace

using namespace rust_compiler::optimizer;

void SummaryWriterPass::runOnOperation() {
  // SummaryWriterPassOptions clOpts{summaryOutputFile.getValue()};
  mlir::ModuleOp module = getOperation();
  //  module.walk([&](mlir::func::FuncOp *f) {
  //  });

  std::string fileName = summaryOutputFile.getValue();
  module.walk([&](rust_compiler::Mir::VTableOp v) {});
}

// https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html
