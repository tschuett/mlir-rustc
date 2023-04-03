#include "Hir/HirDialect.h"
#include "Hir/HirOps.h"
#include "Mir/MirDialect.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace rust_compiler::hir;
using namespace rust_compiler::Mir;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_CONVERTHIRTOMIRPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class ConvertHirToMirPass
    : public rust_compiler::optimizer::impl::ConvertHirToMirPassBase<
          ConvertHirToMirPass> {
public:
  void runOnOperation() override;
};

} // namespace

void ConvertHirToMirPass::runOnOperation() {
  // mlir::ModuleOp m = getOperation();

  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<HirDialect>();
  target.addLegalDialect<MirDialect>();
}
