#include "Lir/LirOps.h"
#include "Lir/LirDialect.h"
#include "Mir/MirDialect.h"
#include "Optimizer/Passes.h"
#include <mlir/Support/LLVM.h>

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Transforms/DialectConversion.h>

#include <mlir/Dialect/Arith/IR/Arith.h>

using namespace rust_compiler::Lir;
using namespace rust_compiler::Mir;

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
  mlir::ModuleOp m = getOperation();

   mlir::ConversionTarget target(getContext());

   target.addIllegalDialect<MirDialect>();
   target.addLegalDialect<LirDialect>();
}
