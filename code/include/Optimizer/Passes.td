include "mlir/Pass/PassBase.td"


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