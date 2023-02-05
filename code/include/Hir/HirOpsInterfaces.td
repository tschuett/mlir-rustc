#ifndef HirInterfaces
#define HirInterfaces

include "mlir/IR/OpBase.td"

class Hir_OpInterface<string name> : OpInterface<name> {
  let cppNamespace = "::rust_compiler::hir";
}

def BorrowOpInterface : Hir_OpInterface<"BorrowOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];
}


#endif
