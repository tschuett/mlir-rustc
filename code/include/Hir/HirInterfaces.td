#ifndef HirInterfaces
#define HirInterfaces

include "mlir/IR/OpBase.td"

def BorrowOpInterface : OpInterface<"BorrowOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];
}

#endif
