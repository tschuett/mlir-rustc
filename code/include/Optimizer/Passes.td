include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def Test : Pass<"test", "mlir::func::FuncOp"> {
  let summary = "Move local array allocations from heap memory into stack memory";
  let description = [{
    Convert heap allocations for arrays, even those of unknown size, into stack
    allocations.
  }];
  let constructor = "rust_compiler::optimizer::createTestPass()";
}


def AttributerLite : Pass<"attributer-lite", "mlir::ModuleOp"> {
  let summary = "Attributer lite";
  let description = [{
    Try to derive some attributes.
  }];
  let constructor = "rust_compiler::optimizer::createAttributerPass()";
}


def RewritePass : Pass<"rewrite", "mlir::func::FuncOp"> {
  let summary = "...";

  // Inherit the common pattern rewrite options from `RewritePassUtils`.
  let options = RewritePassUtils.options;
  let constructor = "rust_compiler::optimizer::createRewriterPass()";
}

def LowerAwaitPass : Pass<"LowerAwait", "mlir::ModuleOp"> {
  let summary = "...";

  let constructor = "rust_compiler::optimizer::createLowerAwaitPass()";
}

def LowerErrorPropagationPass : Pass<"ErrorPropagation", "mlir::func::FuncOp"> {
  let summary = "...";

  let constructor = "rust_compiler::optimizer::createLowerErrorPropagationPass()";
}

def LowerUtilsToLLVMPass : Pass<"UtilsToLLVM", "mlir::ModuleOp"> {
  let summary = "...";

  let constructor = "rust_compiler::optimizer::createLowerUtilsToLLWVMPass()";
}

def SummaryWriterPass : Pass<"SummaryWriter", "mlir::ModuleOp"> {
  let summary = "...";

  let constructor = "rust_compiler::optimizer::createSummaryWriterPass()";
}
