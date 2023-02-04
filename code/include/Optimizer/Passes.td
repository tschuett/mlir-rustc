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
  let summary = "Attributer";
  let description = [{Try to derive some attributes on Mir.}];
}

def RewritePass : Pass<"rewrite", "mlir::func::FuncOp"> {
  let summary = "rewrite Mir to ...";

  // Inherit the common pattern rewrite options from `RewritePassUtils`.
//  let options = RewritePassUtils.options;
}

def LowerAwaitPass : Pass<"LowerAwait", "mlir::ModuleOp"> {
  let summary = "Rust await to async dialect";
  let dependentDialects = [
    "mlir::async::AsyncDialect",
  ];
}

def LowerErrorPropagationPass : Pass<"ErrorPropagation", "mlir::func::FuncOp"> {
  let summary = "...";

}

def GVNPass : Pass<"GVN", "mlir::func::FuncOp"> {
  let summary = "A Sparse Algorithm for Predicated Global Value Numbering from Karthik Gargi.";

}

def DeadCodeEliminationPass : Pass<"dead code elimination", "mlir::func::FuncOp"> {
  let summary = "Deletes dead code..";
}

def DeadStoreEliminationPass : Pass<"dead store elimination", "mlir::ModuleOp"> {
  let summary = "Deletes dead stores..";
}

def SummaryWriterPass : Pass<"SummaryWriter", "mlir::ModuleOp"> {
  let summary = "Write information summary file for ThinLTO";


  let options = [Option<"summaryOutputFile", "summary-output-file", "std::string", "",
                        "Summary Output File">];
}

def SCCPPass : Pass<"SCCP", "mlir::func::FuncOp"> {
  let summary = "Constant propagation with conditional branches";
  let description = [{Wegman, Mark N and Zadeck, F Kenneth}];
}


// eleminate Mir: how?

