#ifndef MirOps
#define MirOps


include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "MirDialect.td"
include "MirTypes.td"
include "MirAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "MirInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

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

def AwaitOp : Mir_Op<"await", []> {
  let summary = "a Rust await op.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$lhs);
}

def ErrorPropagationOp : Mir_Op<"question_mark", []> {
  let summary = "a Rust ? op.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$lhs);
}

//===----------------------------------------------------------------------===//
// AddIOp
//===----------------------------------------------------------------------===//

def AddIOp : Mir_Op<"addi", [Pure]> {
  let summary = "integer addition operation";
  let description = [{
    The shapes of the operands are expected to match.
  }];

  let arguments = (ins AnyInteger:$lhs, AnyInteger:$rhs);
  let results = (outs AnyInteger);

//  // Indicate that the operation has a custom parser and printer method.
//  let hasCustomAssemblyFormat = 1;
//
//  // Allow building an AddiOp with from the two input operands.
//  let builders = [
//    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
//  ];
}

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

def StructAccessOp : Mir_Op<"struct_access", [Pure]> {
  let summary = "struct access";
  let description = [{
    Access the Nth element of a value returning a struct type.
  }];

  let arguments = (ins Mir_StructType:$input, I64Attr:$index);
  let results = (outs AnyType:$output);

  let assemblyFormat = [{
    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
  }];

  // Allow building a StructAccessOp with just a struct value and an index.
  let builders = [
    OpBuilder<(ins "mlir::Value":$input, "size_t":$index)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // Set the folder bit so that we can fold constant accesses.
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// StructConstantOp
//===----------------------------------------------------------------------===//

def StructConstantOp : Mir_Op<"struct_constant", [ConstantLike, Pure]> {
  let summary = "struct constant";
  let description = [{
    Constant operation turns a literal struct value into an SSA value. The data
    is attached to the operation as an attribute. The struct constant is encoded
    as an array of other constant values. For example:

    ```mlir
      %0 = Mir.struct_constant [
        dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
      ] : !Mir.struct<tensor<*xf64>>
    ```
  }];

  let arguments = (ins ArrayAttr:$value);
  let results = (outs Mir_StructType:$output);

  let assemblyFormat = "$value attr-dict `:` type($output)";

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
  let hasFolder = 1;
}

#endif
