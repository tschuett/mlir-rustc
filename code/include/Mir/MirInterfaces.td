#ifndef MirInterfaces
#define MirInterfaces


include "mlir/IR/OpBase.td"

def BorrowOpInterface : OpInterface<"BorrowOpInterface"> {
  let description = [{
    This is an example interface definition.
  }];
}

#endif
