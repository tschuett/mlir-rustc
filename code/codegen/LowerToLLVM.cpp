#include "CodeGen/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

using namespace mlir;

namespace rust_compiler::codegen {
#define GEN_PASS_DEF_MIRTOLLVMLOWERING
#include "CodeGen/Passes.h.inc"
} // namespace rust_compiler::codegen

namespace {
struct MirToLLVMLoweringPass
    : public rust_compiler::codegen::impl::MirToLLVMLoweringBase<
          MirToLLVMLoweringPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void MirToLLVMLoweringPass::runOnOperation() {
  // FIXME
}

// namespace rust_compiler
