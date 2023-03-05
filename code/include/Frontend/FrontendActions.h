#pragma once

#include "Frontend/FrontendAction.h"

#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::frontend {

class SyntaxAonly : public FrontendAction {};

class SemaOnylAction : public FrontendAction {};

class CodeGenAction : public FrontendAction {
  void executeAction() override;
  /// Runs prescan, parsing, sema and lowers to MLIR.
  bool beginSourceFileAction() override;
  /// Sets up LLVM's TargetMachine.
  void setUpTargetMachine();
  /// Runs the optimization (aka middle-end) pipeline on the LLVM module
  /// associated with this action.
  void runOptimizationPipeline(llvm::raw_pwrite_stream &os);

  std::unique_ptr<mlir::ModuleOp> mlirModule;
  std::unique_ptr<mlir::MLIRContext> mlirCtx;

  void generateLLVMIR();

  std::unique_ptr<llvm::TargetMachine> tm;
};

} // namespace rust_compiler::frontend
