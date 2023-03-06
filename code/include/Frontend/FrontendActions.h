#pragma once

#include "Frontend/FrontendAction.h"

#include <llvm/IR/Module.h>
#include <llvm/Target/TargetMachine.h>
#include <mlir/IR/BuiltinOps.h>

namespace rust_compiler::frontend {

class SyntaxOnlyAction : public FrontendAction {
public:
  virtual ~SyntaxOnlyAction() = default;

  void executeAction() override;
};

class SemaOnlylAction : public FrontendAction {
public:
  virtual ~SemaOnlylAction() = default;

  void executeAction() override;
};

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

  /// @name LLVM IR
  std::unique_ptr<llvm::LLVMContext> llvmCtx;
  std::unique_ptr<llvm::Module> llvmModule;

  std::unique_ptr<llvm::TargetMachine> tm;

  /// Generates an LLVM IR module from CodeGenAction::mlirModule and saves it
  /// in CodeGenAction::llvmModule.
  void generateLLVMIR();

  void generateObjectFile(llvm::raw_pwrite_stream &os);

  void setMLIRDataLayout(mlir::ModuleOp &mlirModule,
                         const llvm::DataLayout &dl);
};

} // namespace rust_compiler::frontend
