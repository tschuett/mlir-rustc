#include "Lir/LirDialect.h"
#include "Lir/LirOps.h"
#include "Mir/MirDialect.h"
#include "Optimizer/Passes.h"
#include <mlir/Dialect/MemRef/IR/MemRef.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
//#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
//#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

using namespace rust_compiler::Lir;
using namespace rust_compiler::Mir;
using namespace mlir;
using namespace llvm;

//===----------------------------------------------------------------------===//
// MirToLirPass
//===----------------------------------------------------------------------===//

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_CONVERTMIRTOLIRPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class ConvertMirToLirPass
    : public rust_compiler::optimizer::impl::ConvertMirToLirPassBase<
          ConvertMirToLirPass> {
public:
  void runOnOperation() override;
};

} // namespace

void ConvertMirToLirPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addLegalOp<mlir::ModuleOp>();

  target.addIllegalDialect<MirDialect>();

  target.addLegalDialect<LirDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalDialect<mlir::func::FuncDialect>();

  RewritePatternSet patterns(&getContext());
  //patterns.add<PrintOpLowering>(&getContext());

  mlir::ModuleOp module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
