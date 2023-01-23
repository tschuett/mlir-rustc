#include "CodeGen/Passes.h"
#include "Mir/MirOps.h"

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

namespace rust_compiler::codegen {
#define GEN_PASS_DEF_LOWERUTILSTOLLVMPASS
#include "CodeGen/Passes.h.inc"
} // namespace rust_compiler::codegen

namespace {
class LowerUtilsToLLVMPass
  : public rust_compiler::codegen::impl::LowerUtilsToLLVMPassBase<
          LowerUtilsToLLVMPass> {
public:
  void runOnOperation() override;
};

} // namespace

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
