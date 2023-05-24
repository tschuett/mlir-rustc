#include "Optimizer/Passes.h"
#include "Mir/MirOps.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h>
#include <mlir/Dialect/Async/IR/Async.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Operation.h>

using namespace mlir;

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_LOWERUTILSTOLLVMPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class LowerUtilsToLLVMPass
    : public rust_compiler::optimizer::impl::LowerUtilsToLLVMPassBase<
          LowerUtilsToLLVMPass> {
public:
  void runOnOperation() override;
};

} // namespace

void LowerUtilsToLLVMPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(converter, patterns);
  mlir::cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  mlir::populateFinalizeMemRefToLLVMConversionPatterns(converter, patterns);
  mlir::populateFuncToLLVMConversionPatterns(converter, patterns);
  //mlir::populateFuncOpToLLVMConversionPattern(converter, patterns);
  mlir::populateVectorToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
