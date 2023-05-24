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
  let options = RewritePassUtils.options;
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

def SCCPPass : Pass<"SCCP", "mlir::ModuleOp"> {
  let summary = "Constant propagation with conditional branches";
  let description = [{Wegman, Mark N and Zadeck, F Kenneth}];
}

def HirLICMPass : Pass<"SCCP", "mlir::func::FuncOp"> {
  let summary = "LICM on Hir loops";
  let description = "Loop-Invariant Computation and Code Motion on Hir Loops";
}


def ConvertHirToMirPass : Pass<"Hir2Mir", "mlir::ModuleOp"> {
  let summary = "Hir -> Mir";
  let description = "Convert Hir to Mir dialect";
  let dependentDialects = ["::MirDialect", "::mlir::arith::ArithDialect"];
}

def ConvertMirToLirPass : Pass<"Mir2Lir", "mlir::ModuleOp"> {
  let summary = "Hir -> Mir";
  let description = "Convert Mir to Lir dialect";
  let dependentDialects = ["::LirDialect", "::mlir::arith::ArithDialect"];
}

def ConvertLirToLLVMPass : Pass<"Lir2LLVM", "mlir::ModuleOp"> {
  let summary = "Lir -> LLVM";
  let description = "Convert Lir to LLVM dialect";
  let dependentDialects = ["::LirDialect", "::mlir::arith::ArithDialect"];
}

def LoopPass : Pass<"Loop", "mlir::func::FuncOp"> {
  let summary = "Fat loops";
  let description = "Fat loop pass";
}

def DeadArgumentEliminationPass : Pass<"DeadArgumentElimination", "mlir::ModuleOp"> {
  let summary = "Eliminates dead arguments";
  let description = "IPO dead argument elimination";
}

def FuncSpecialPass : Pass<"FuncSpecial", "mlir::ModuleOp"> {
  let summary = "specializes funcs";
  let description = "specializes funcs after SCCP";
}

def MemoryAnalysisPass : Pass<"MemoryAnalysis", "mlir::ModuleOp"> {
  let summary = "memory analysis";
  let description = "analyses memrefs";
}

def LowerUtilsToLLVMPass : Pass<"LowerUtilsToLLVM", "mlir::ModuleOp"> {
  let summary = "lower dialects to LLVM";
  let description = "lower dialects to LLVM";
}



// eleminate Mir: how?

