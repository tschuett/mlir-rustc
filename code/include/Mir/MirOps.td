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
  let hasVerifier = 1;

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

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getMethodAttrNameStr() { return "method"; }
    static constexpr llvm::StringRef getProcAttrNameStr() { return "proc"; }
  }];
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

def Mir_ConstantOp : Op<Mir_Dialect, "constant",
    [ConstantLike, Pure, AllTypesMatch<["value", "result"]>]> {
  let summary = "integer or floating point constant";
  let description = [{
    The `constant` operation produces an SSA value equal to some integer or
    floating-point constant specified by an attribute. This is the way MLIR
    forms simple integer and floating point constants.

    Example:

    ```
    // Integer constant
    %1 = Mir.constant 42 : i32

    // Equivalent generic form
    %1 = "Mir.constant"() {value = 42 : i32} : () -> i32
    ```
  }];

  let arguments = (ins TypedAttrInterface:$value);
  // TODO: Disallow Mir.constant to return anything other than a signless
  // integer or float like. Downstream users of Mir should only be
  // working with signless integers, floats, or vectors/tensors thereof.
  // However, it is necessary to allow arith.constant to return vectors/tensors
  // of strings and signed/unsigned integers (for now) as an artefact of
  // splitting the Standard dialect.
  let results = (outs /*SignlessIntegerOrFloatLike*/AnyType:$result);

  let builders = [
    OpBuilder<(ins "mlir::Attribute":$value, "mlir::Type":$type),
    [{ build($_builder, $_state, type, value); }]>,
  ];

  let extraClassDeclaration = [{
    /// Whether the constant op can be constructed with a particular value and
    /// type.
    static bool isBuildableWith(mlir::Attribute value, mlir::Type type);
  }];

  let assemblyFormat = "attr-dict $value";
}

//===----------------------------------------------------------------------===//
// CondBranchOp
//===----------------------------------------------------------------------===//

def CondBranchOp : Mir_Op<"cond_br",
    [AttrSizedOperandSegments,
     DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
     Pure, Terminator]> {
  let summary = "conditional branch operation";
  let description = [{
    The `cond_br` terminator operation represents a conditional branch on a
    boolean (1-bit integer) value. If the bit is set, then the first destination
    is jumped to; if it is false, the second destination is chosen. The count
    and types of operands must align with the arguments in the corresponding
    target blocks.

    The MLIR conditional branch operation is not allowed to target the entry
    block for a region. The two destinations of the conditional branch operation
    are allowed to be the same.

    The following example illustrates a function with a conditional branch
    operation that targets the same block.

    Example:

    ```mlir
    func.func @select(%a: i32, %b: i32, %flag: i1) -> i32 {
      // Both targets are the same, operands differ
      cond_br %flag, ^bb1(%a : i32), ^bb1(%b : i32)

    ^bb1(%x : i32) :
      return %x : i32
    }
    ```
  }];


  let arguments = (ins I1:$condition,
                       Variadic<AnyType>:$trueDestOperands,
                       Variadic<AnyType>:$falseDestOperands);
  let successors = (successor AnySuccessor:$trueDest, AnySuccessor:$falseDest);

  let builders = [
    OpBuilder<(ins "mlir::Value":$condition, "mlir::Block *":$trueDest,
      "mlir::ValueRange":$trueOperands, "mlir::Block *":$falseDest,
      "mlir::ValueRange":$falseOperands), [{
      build($_builder, $_state, condition, trueOperands, falseOperands, trueDest,
            falseDest);
    }]>,
    OpBuilder<(ins "mlir::Value":$condition, "mlir::Block *":$trueDest,
      "mlir::Block *":$falseDest, CArg<"mlir::ValueRange", "{}">:$falseOperands), [{
      build($_builder, $_state, condition, trueDest, mlir::ValueRange(), falseDest,
            falseOperands);
    }]>];

  let extraClassDeclaration = [{
    // These are the indices into the dests list.
    enum { trueIndex = 0, falseIndex = 1 };

    // Accessors for operands to the 'true' destination.
    mlir::Value getTrueOperand(unsigned idx) {
      assert(idx < getNumTrueOperands());
      return getOperand(getTrueDestOperandIndex() + idx);
    }

    void setTrueOperand(unsigned idx, mlir::Value value) {
      assert(idx < getNumTrueOperands());
      setOperand(getTrueDestOperandIndex() + idx, value);
    }

    unsigned getNumTrueOperands()  { return getTrueOperands().size(); }

    /// Erase the operand at 'index' from the true operand list.
    void eraseTrueOperand(unsigned index)  {
      getTrueDestOperandsMutable().erase(index);
    }

    // Accessors for operands to the 'false' destination.
    mlir::Value getFalseOperand(unsigned idx) {
      assert(idx < getNumFalseOperands());
      return getOperand(getFalseDestOperandIndex() + idx);
    }
    void setFalseOperand(unsigned idx, mlir::Value value) {
      assert(idx < getNumFalseOperands());
      setOperand(getFalseDestOperandIndex() + idx, value);
    }

    operand_range getTrueOperands() { return getTrueDestOperands(); }
    operand_range getFalseOperands() { return getFalseDestOperands(); }


    unsigned getNumFalseOperands() { return getFalseOperands().size(); }

    /// Erase the operand at 'index' from the false operand list.
    void eraseFalseOperand(unsigned index) {
      getFalseDestOperandsMutable().erase(index);
    }


  private:
    /// Get the index of the first true destination operand.
    unsigned getTrueDestOperandIndex() { return 1; }

    /// Get the index of the first false destination operand.
    unsigned getFalseDestOperandIndex() {
      return getTrueDestOperandIndex() + getNumTrueOperands();
    }
  }];

  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueDestOperands^ `:` type($trueDestOperands) `)`)? `,`
    $falseDest (`(` $falseDestOperands^ `:` type($falseDestOperands) `)`)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// BranchOp
//===----------------------------------------------------------------------===//

def BranchOp : Mir_Op<"br", [
    DeclareOpInterfaceMethods<BranchOpInterface, ["getSuccessorForOperands"]>,
    Pure, Terminator
  ]> {
  let summary = "branch operation";
  let description = [{
    The `Mir.br` operation represents a direct branch operation to a given
    block. The operands of this operation are forwarded to the successor block,
    and the number and type of the operands must match the arguments of the
    target block.

    Example:

    ```mlir
    ^bb2:
      %2 = call @someFn()
      Mir.br ^bb3(%2 : tensor<*xf32>)
    ^bb3(%3: tensor<*xf32>):
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);

  let builders = [
    OpBuilder<(ins "mlir::Block *":$dest,
                   CArg<"mlir::ValueRange", "{}">:$destOperands), [{
      $_state.addSuccessors(dest);
      $_state.addOperands(destOperands);
    }]>];

  let extraClassDeclaration = [{
    void setDest(mlir::Block *block);

    /// Erase the operand at 'index' from the operand list.
    void eraseOperand(unsigned index);
  }];

  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// SelectOp
//===----------------------------------------------------------------------===//

def SelectOp : Mir_Op<"select", [Pure,
    AllTypesMatch<["true_value", "false_value", "result"]>
  ] # ElementwiseMappable.traits> {
  let summary = "select operation";
  let description = [{
    The `Mir.select` operation chooses one value based on a binary condition
    supplied as its first operand. If the value of the first operand is `1`,
    the second operand is chosen, otherwise the third operand is chosen.
    The second and the third operand must have the same type.

    The operation applies to vectors and tensors elementwise given the _shape_
    of all operands is identical. The choice is made for each element
    individually based on the value at the same position as the element in the
    condition operand. If an i1 is provided as the condition, the entire vector
    or tensor is chosen.

    Example:

    ```mlir
    // Custom form of scalar selection.
    %x = Mir.select %cond, %true, %false : i32

    // Generic form of the same operation.
    %x = "Mir.select"(%cond, %true, %false) : (i1, i32, i32) -> i32

    // Element-wise vector selection.
    %vx = Mir.select %vcond, %vtrue, %vfalse : vector<42xi1>, vector<42xf32>

    // Full vector selection.
    %vx = Mir.select %cond, %vtrue, %vfalse : vector<42xf32>
    ```
  }];

  let arguments = (ins BoolLike:$condition,
                       AnyType:$true_value,
                       AnyType:$false_value);
  let results = (outs AnyType:$result);

}

#endif
