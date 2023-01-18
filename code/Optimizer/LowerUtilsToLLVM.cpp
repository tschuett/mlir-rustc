#include "Mir/MirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
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
  LowerUtilsToLLVMPass() = default;
  LowerUtilsToLLVMPass(const LowerUtilsToLLVMPass &pass);

  llvm::StringRef getDescription() const override;

  void runOnOperation() override;
};

} // namespace

LowerUtilsToLLVMPass::LowerUtilsToLLVMPass(const LowerUtilsToLLVMPass &pass)
    : rust_compiler::optimizer::impl::LowerUtilsToLLVMPassBase<
          LowerUtilsToLLVMPass>(pass) {}

llvm::StringRef LowerUtilsToLLVMPass::getDescription() const {
  return "test pass";
}

void LowerUtilsToLLVMPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  RewritePatternSet patterns(&getContext());
  LLVMTypeConverter converter(&getContext());
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  populateMemRefToLLVMConversionPatterns(converter, patterns);
  populateFuncToLLVMConversionPatterns(converter, patterns);

  LLVMConversionTarget target(getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> createUtilsToLLVMPass() {
  return std::make_unique<LowerUtilsToLLVMPass>();
}
