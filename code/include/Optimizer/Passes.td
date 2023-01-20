include "mlir/Pass/PassBase.td"
include "mlir/Rewrite/PassUtil.td"

def Test : Pass<"test", "mlir::func::FuncOp"> {
  let summary =
      "Move local array allocations from heap memory into stack memory";
  let description = [{
    Convert heap allocations for arrays, even those of unknown size, into stack
    allocations.
  }];
}

def Attributer : Pass<"attributer", "mlir::ModuleOp"> {
  let summary = "Attributer lite";
  let description = [{Try to derive some attributes on Mir.}];
}

def RewritePass : Pass<"rewrite", "mlir::func::FuncOp"> {
  let summary = "rewrite Mir to ...";

  // Inherit the common pattern rewrite options from `RewritePassUtils`.
  let options = RewritePassUtils.options;
  let constructor = "rust_compiler::optimizer::createRewriterPass()";
}

def LowerAwaitPass : Pass<"LowerAwait", "mlir::ModuleOp"> {
  let summary = "Rust await to async dialect";

}

def LowerErrorPropagationPass : Pass<"ErrorPropagation", "mlir::func::FuncOp"> {
  let summary = "...";

}

def LowerUtilsToLLVMPass : Pass<"UtilsToLLVM", "mlir::ModuleOp"> {
  let summary = "lower: arith, cf, memref, and func to LLVM";

}

def GVNPass : Pass<"GVN", "mlir::func::FuncOp"> {
  let summary = "A Sparse Algorithm for Predicated Global Value Numbering from Karthik Gargi.";

}

def SummaryWriterPass : Pass<"SummaryWriter", "mlir::ModuleOp"> {
  let summary = "Write information summary file for ThinLTO";


  let options = [Option<"summaryOutputFile", "summary-output-file", "std::string", "",
                        "Summary Output File">];
}




// eleminate Mir: how?

