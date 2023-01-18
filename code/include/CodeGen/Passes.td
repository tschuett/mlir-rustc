include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def MirToLLVMLowering : Pass<"Mir to LLVM", "mlir::func::FuncOp"> {
  let summary = "";
  let description = [{
  }];
  let constructor = "rust_compiler::codegen::createLowerMirToLLVMPass()";
}

