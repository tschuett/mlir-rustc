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

  let results = (outs AnyType:$output);
}

def MutBorrowOp : Mir_Op<"mutborrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a mutable borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);

  let results = (outs AnyType:$output);
}

def AwaitOp : Mir_Op<"await", []> {
  let summary = "a Rust await op.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$lhs);

  let results = (outs AnyType:$output);
}

//===----------------------------------------------------------------------===//
// VarDeclaration
//===----------------------------------------------------------------------===//

def VarDeclarationOp : Mir_Op<"vardeclare", []> {
  let summary = "Auxilliary  information for variable declarations.";

  let description = [{
  Based on the fir.declare concept. It takes as input an alloc and provides
  additional information.
  }];

  let arguments = (ins
    AnyType:$memref,
    Builtin_StringAttr:$uniq_name
  );

  let results = (outs AnyType:$output);
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

//  // Indicate that additional verification for this operation is necessary.
//  let hasVerifier = 1;
//
//  // Set the folder bit so that we can fold constant accesses.
//  let hasFolder = 1;
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
//  let hasVerifier = 1;
//  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

def Mir_MirEndOp : Mir_Op<"end", [Terminator]> {
  let summary = "the end instruction";

  let description = [{
    The end terminator is a special terminator used inside various MIR
    operations that have regions.  End is thus the custom invisible terminator
    for these operations.  It is implicit and need not appear in the textual
    representation.
  }];
}

def ImplicitMirTerminator : SingleBlockImplicitTerminator<"MirEndOp">;


def VTableOp: Mir_Op<"vtable", [IsolatedFromAbove, Symbol, ImplicitMirTerminator]> {
  let summary = "Vtable";

  let description = [{
   ```mlir
      mir.vtable @_QDTMquuzTfoo {
        mir.vt_entry method1, @_QFNMquuzTfooPmethod1AfooR
        mir.vt_entry method2, @_QFNMquuzTfooPmethod2AfooII
      }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$parent
  );

  let hasCustomAssemblyFormat = 1;

  let regions = (region AnyRegion:$region);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "llvm::StringRef":$name, "mlir::Type":$type,
      "llvm::StringRef":$parent,
      CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getParentAttrNameStr() { return "parent"; }
    static constexpr llvm::StringRef getExtendsKeyword() { return "extends"; }

    mlir::Block &getBlock() {
      return getRegion().front();
    }
  }];
}

def VTEntryOp : Mir_Op<"vt_entry", [HasParent<"VTableOp">]> {
  let summary = "map entry in a vtable";

  let description = [{
    An entry in a vtable.  Allows a function symbol to be bound
    to a specifier method identifier.  A dispatch operation uses the dynamic
    type of a distinguished argument to determine an exact dispatch table
    and uses the method identifier to select the type-bound procedure to
    be called.

    ```mlir
      mir.vt_entry method_name, @uniquedProcedure
    ```
  }];

  let arguments = (ins StrAttr:$method, SymbolRefAttr:$proc);

  //let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getMethodAttrNameStr() { return "method"; }
    static constexpr llvm::StringRef getProcAttrNameStr() { return "proc"; }
  }];
}

//===----------------------------------------------------------------------===//
// ClosureOp
//===----------------------------------------------------------------------===//

def ClosureOp : Mir_Op<"closure", []> {

  let arguments = (ins BoolLike:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);

   let builders = [
    OpBuilder<(ins CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>
  ];

  let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// ClosureMoveOp
//===----------------------------------------------------------------------===//

def ClosureMoveOp : Mir_Op<"closure_move", []> {

  let arguments = (ins BoolLike:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);

   let builders = [
    OpBuilder<(ins CArg<"llvm::ArrayRef<mlir::NamedAttribute>", "{}">:$attributes)>
  ];

  let regions = (region AnyRegion:$region);
}

#endif
