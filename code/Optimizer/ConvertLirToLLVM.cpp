#include "Lir/LirDialect.h"
#include "Lir/LirOps.h"
#include "Optimizer/Passes.h"

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>
// #include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
// #include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Conversion/LLVMCommon/ConversionTarget.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Conversion/ArithToLLVM/ArithToLLVM.h>
#include <mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h>
#include <mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h>
#include <mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h>

using namespace rust_compiler::Lir;
using namespace rust_compiler::optimizer;
using namespace llvm;
using namespace mlir;

//===----------------------------------------------------------------------===//
// LirToLLLVMPass
//===----------------------------------------------------------------------===//

namespace rust_compiler::optimizer {
#define GEN_PASS_DEF_CONVERTLIRTOLLVMPASS
#include "Optimizer/Passes.h.inc"
} // namespace rust_compiler::optimizer

namespace {
class ConvertLirToLLVMPass
    : public rust_compiler::optimizer::impl::ConvertLirToLLVMPassBase<
          ConvertLirToLLVMPass> {
public:
  void runOnOperation() override;
};

} // namespace

void ConvertLirToLLVMPass::runOnOperation() {
  LLVMConversionTarget target(getContext());

  target.addLegalOp<mlir::ModuleOp>();

  //target.addIllegalDialect<LirDialect>();
  //target.addLegalDialect<mlir::memref::MemRefDialect>();
  //target.addLegalDialect<mlir::arith::ArithDialect>();
  //target.addLegalDialect<mlir::func::FuncDialect>();

  LLVMTypeConverter typeConverter(&getContext());

  RewritePatternSet patterns(&getContext());
  mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  // patterns.add<PrintOpLowering>(&getContext());

  mlir::ModuleOp module = getOperation();
  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
