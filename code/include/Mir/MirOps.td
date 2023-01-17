#ifndef MirOps
#define MirOps


include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "MirDialect.td"
include "MirTypes.td"
include "MirAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "MirInterfaces.td"


// Ops
class Mir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mir_Dialect, mnemonic, traits>;

def BorrowOp : Mir_Op<"borrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins AnyType:$rhs);
}

def MutBorrowOp : Mir_Op<"mutborrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a mutable borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);
}

def AwaitOp : Mir_Op<"await", [Pure]> {
  let summary = "a Rust await op.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);
}


#endif
