include "mlir/Pass/PassBase.td"

def MirToLLVMLowering : Pass<"Mir to LLVM", "mlir::func::FuncOp"> {
  let summary = "";
  let description = [{
  }];
  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
}

def LowerUtilsToLLVMPass : Pass<"UtilsToLLVM", "mlir::ModuleOp"> {
  let summary = "lower: arith, cf, memref, and func to LLVM";
  let dependentDialects = [
    "LLVM::LLVMDialect",
  ];
}


