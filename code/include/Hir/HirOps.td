#ifndef HirOps
#define HirOps

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/IR/BuiltinAttributes.td"
include "HirDialect.td"
include "HirOpsBase.td"
include "HirTypes.td"
include "HirPattern.td"
include "HirAttr.td"
include "HirOpsInterfaces.td"

// Ops
class Hir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Hir_Dialect, mnemonic, traits>;


def ErrorPropagationOp : Hir_Op<"question_mark", []> {
  let summary = "a Rust ? op.";

  let description = [{
    Rust ErrorPropagation Expression
  }];

  let arguments = (ins AnyType:$rhs);

  let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// PredicateLoopExpression
//===----------------------------------------------------------------------===//

def PredicateLoopExpressionOp : Hir_Op<"while",
  [LoopLikeOpInterface]> {
  let summary = "a Rust while op.";

  let description = [{
  Rust PredicateLoopExpressionn
  }];

  let arguments = (ins
    I1:$rhs
  );

  let extraClassDeclaration = [{
    mlir::Value getConditionValue() { return getRhs(); }

    mlir::Region& getLoopBody()  { return getRegion();}
  }];


  let regions = (region AnyRegion:$region);
}

//===----------------------------------------------------------------------===//
// InfiniteLoopExpression
//===----------------------------------------------------------------------===//

def InfiniteLoopOp : Hir_Op<"loop",
  [LoopLikeOpInterface]> {
  let summary = "a Rust loop op.";

  let description = [{
  Rust InfiniteLoopExpressionn
  }];

  let extraClassDeclaration = [{
    mlir::Region& getLoopBody()  { return getRegion();}
  }];

  let regions = (region AnyRegion:$region);
}

def BreakOp : Hir_Op<"break",
  []> {
  let summary = "a Rust break op.";

  let description = [{
  Rust break expression
  }];

  let arguments = (ins AnyType:$rhs);
}

def IfOp : Hir_Op<"if", [NoRegionArguments]> {
  let summary = "a Rust if op.";

  let description = [{
  Rust IfExpression
  }];

  let arguments = (ins I1:$condition);

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region
    SizedRegion<1>:$thenRegion,
    AnyRegion:$elseRegion
  );
}

def ArrayOp : Hir_Op<"array",
  []> {
  let summary = "a ArrayExpression op.";

  let description = [{
  Rust ArrayExpression
  }];

  let arguments = (ins Variadic<AnyType>:$els);

  let results = (outs AnyType);
}

def AwaitOp : Hir_Op<"await",
  []> {
  let summary = "a AwaitExpression op.";

  let description = [{
  Rust AwaitExpression
  }];

  let arguments = (ins AnyType:$future);

  let results = (outs AnyType);
}

def BlockOp : Hir_Op<"block",
  []> {
  let summary = "a AwaitExpression op.";

  let description = [{
  Rust AwaitExpression
  }];

  //let arguments = (ins AnyType:$future);

  let regions = (region AnyRegion:$region);

  let results = (outs AnyType);
}

def UnsafeBlockOp : Hir_Op<"unsafe block",
  []> {
  let summary = "a UnsafeExpression op.";

  let description = [{
  Rust UnsafeExpression
  }];

  //let arguments = (ins AnyType:$future);

  let regions = (region AnyRegion:$region);

  let results = (outs AnyType);
}

//===----------------------------------------------------------------------===//
// &
//===----------------------------------------------------------------------===//

def BorrowOp : Hir_Op<"borrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins AnyType:$rhs);

  let results = (outs AnyType:$output);
}

//===----------------------------------------------------------------------===//
// &mut
//===----------------------------------------------------------------------===//

def MutBorrowOp : Hir_Op<"mutborrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a mutable borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);

  let results = (outs AnyType:$output);
}

//===----------------------------------------------------------------------===//
// VarDeclaration
//===----------------------------------------------------------------------===//

// AttrSizedOperandSegments

def Hir_DeclareOp : Hir_Op<"declare", []> {
  let description = [{
                 declare a variable and produce an
                 SSA value that can be used as a
                 variable in Hir operations
  }];

  let arguments = (ins
    Builtin_StringAttr:$uniq_name
  );

  let results = (outs AnyType);
}

//===----------------------------------------------------------------------===//
// VarDeclaration helper
//===----------------------------------------------------------------------===//

def VarDeclarationOp : Hir_Op<"vardeclare", []> {
  let summary = "Auxilliary  information for variable declarations.";

  let description = [{
  Based on the hir.declare concept. It takes as input an alloc and provides
  additional information.
  }];

  let arguments = (ins
    AnyType:$vardecl,
    Builtin_StringAttr:$uniq_name
  );

  let results = (outs AnyType:$output);
}

//===----------------------------------------------------------------------===//
// VTable
//===----------------------------------------------------------------------===//

def Hir_HirEndOp : Hir_Op<"end", [Terminator]> {
  let summary = "the end instruction";

  let description = [{
    The end terminator is a special terminator used inside various HIR
    operations that have regions.  End is thus the custom invisible terminator
    for these operations.  It is implicit and need not appear in the textual
    representation.
  }];
}

def ImplicitHirTerminator : SingleBlockImplicitTerminator<"HirEndOp">;


def VTableOp: Hir_Op<"vtable", [IsolatedFromAbove, Symbol, ImplicitHirTerminator]> {
  let summary = "Vtable";

  let description = [{
   ```mlir
      hir.vtable @_QDTMquuzTfoo {
        hir.vt_entry method1, @_QFNMquuzTfooPmethod1AfooR
        hir.vt_entry method2, @_QFNMquuzTfooPmethod2AfooII
      }
    ```
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$parent
  );

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

def VTEntryOp : Hir_Op<"vt_entry", [HasParent<"VTableOp">]> {
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

  let extraClassDeclaration = [{
    static constexpr llvm::StringRef getMethodAttrNameStr() { return "method"; }
    static constexpr llvm::StringRef getProcAttrNameStr() { return "proc"; }
  }];
}

//===----------------------------------------------------------------------===//
// AssignOp
//===----------------------------------------------------------------------===//

def Hir_AssignOp : Hir_Op<"assign", [MemoryEffects<[MemWrite]>]> {
  let summary = "Assign an expression or variable value to a Rust variable";

  let description = [{
    Assign rhs to lhs following Rust assignments rules.
  }];

//  let arguments = (ins AnyFortranEntity:$rhs,
//                   Arg<AnyFortranVariable, "", [MemWrite]>:$lhs);
//
  let arguments = (ins AnyType:$rhs,
                   Arg<AnyType, "", [MemWrite]>:$lhs);

  let assemblyFormat = [{
    $rhs `to` $lhs attr-dict `:` type(operands)
  }];
}

//===----------------------------------------------------------------------===//
// StructAccessOp
//===----------------------------------------------------------------------===//

//def StructAccessOp : Hir_Op<"struct_access", [Pure]> {
//  let summary = "struct access";
//  let description = [{
//    Access the Nth element of a value returning a struct type.
//  }];
//
//  let arguments = (ins Hir_StructType:$input, I64Attr:$index);
//  let results = (outs AnyType:$output);
//
//  let assemblyFormat = [{
//    $input `[` $index `]` attr-dict `:` type($input) `->` type($output)
//  }];
//
//  // Allow building a StructAccessOp with just a struct value and an index.
//  let builders = [
//    OpBuilder<(ins "mlir::Value":$input, "size_t":$index)>
//  ];
//}


//===----------------------------------------------------------------------===//
// StructConstructorOp
//===----------------------------------------------------------------------===//

//def StructConstructorOp : Hir_Op<"struct_constructor", [Pure]> {
//  let summary = "struct constant";
//  let description = [{
//    Constant operation turns a literal struct value into an SSA value. The data
//    is attached to the operation as input values. The struct constant is encoded
//    as an array of other constant values. For example:
//
//    ```mlir
//      %0 = hir.struct_constructor [
//        dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
//      ] : !hir.struct<tensor<*xf64>>
//    ```
//  }];
//
//  let arguments = (ins
//                  Builtin_StringAttr:$path,
//                  Variadic<AnyType>:$value
//                  );
//
//  let results = (outs AnyType:$output);
//
////  let assemblyFormat = "$path $value attr-dict `:` type($output)";
//}


////===----------------------------------------------------------------------===//
//// ConstantOp
////===----------------------------------------------------------------------===//
//
//def ConstantOp : Hir_Op<"constant", [ConstantLike, Pure]> {
//  let summary = "any attribute constant";
//  let description = [{
//    Constant operation turns a literal expression attribute value into an SSA value. The data
//    is attached to the operation as an attribute. For example:
//
//    ```mlir
//      %0 = hir.constant [
//        dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64>
//      ] : !hir.attribute
//    ```
//  }];
//
//  let arguments = (ins AnyAttr:$value);
//  let results = (outs Hir_ConstantType:$output);
//
//  let assemblyFormat = "attr-dict `:` type($output)";
//}

//===----------------------------------------------------------------------===//
// UnitConstantOp
//===----------------------------------------------------------------------===//

def UnitConstantOp : Hir_Op<"unit_constant", [ConstantLike, Pure]> {
  let summary = "unit constant";
  let description = [{
    Constant operation turns a unit value into an SSA value. For example:

    ```mlir
      %0 = hir.unit_constant [
      ] : !hir.unit_constant
    ```
  }];

  //let arguments = (ins Builtin_StringAttr:$value);
  let results = (outs NoneType:$output);

  let assemblyFormat = "attr-dict `:` type($output)";
}

//===----------------------------------------------------------------------===//
// RangeExprOp
//===----------------------------------------------------------------------===//

def RangeExprOp : Hir_Op<"range_expr", [Pure]> {
  let summary = "RangeExpr";
  let description = [{
    Represents a range expression from .. to. For example:

    ```mlir
      %0 = hir.range_expr %0, %1
    ```
  }];

  let arguments = (ins AnyScalarObject:$left, AnyScalarObject:$right);
  let results = (outs AnyType:$output);

  //let assemblyFormat = "$output attr-dict  `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RangeFromExprOp
//===----------------------------------------------------------------------===//

def RangeFromExprOp : Hir_Op<"range_from_expr", [Pure]> {
  let summary = "RangeFromExpr";
  let description = [{
    Represents a range expression  from ... For example:

    ```mlir
      %0 = hir.range_from_expr %1
    ```
  }];

  let arguments = (ins AnyScalarObject:$left);
  let results = (outs AnyType:$output);

  //let assemblyFormat = "$output attr-dict  `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RangeToExprOp
//===----------------------------------------------------------------------===//

def RangeToExprOp : Hir_Op<"range_to_expr", [Pure]> {
  let summary = "RangeToExpr";
  let description = [{
    Represents a range expression  .. to. For example:

    ```mlir
      %0 = hir.range_to_expr %1
    ```
  }];

  let arguments = (ins AnyScalarObject:$left);
  let results = (outs AnyType:$output);

  //let assemblyFormat = "$output attr-dict  `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RangeFullExprOp
//===----------------------------------------------------------------------===//

def RangeFullExprOp : Hir_Op<"range_full_expr", [ConstantLike, Pure]> {
  let summary = "RangeFullExpr";
  let description = [{
    Represents a range expression  .. . For example:

    ```mlir
      %0 = hir.range_full_expr
    ```
  }];

  //let arguments = (ins AnyScalarObject:$left);
  let results = (outs AnyType:$output);

  //let assemblyFormat = "$output attr-dict  `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RangeInclusiveExprOp
//===----------------------------------------------------------------------===//

def RangeInclusiveExprOp : Hir_Op<"range_inclusive_expr", [Pure]> {
  let summary = "RangeInclusiveExpr";
  let description = [{
    Represents a range expression  from ..= to . For example:

    ```mlir
      %0 = hir.range_inclusive_expr %1, %2
    ```
  }];

  let arguments = (ins AnyScalarObject:$left, AnyScalarObject:$right);
  let results = (outs AnyType:$output);

  //let assemblyFormat = "$output attr-dict  `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// RangeToInclusiveExprOp
//===----------------------------------------------------------------------===//

def RangeToInclusiveExprOp : Hir_Op<"range_to_inclusive_expr", [Pure]> {
  let summary = "RangeToInclusiveExpr";
  let description = [{
    Represents a range expression   ..= to . For example:

    ```mlir
      %0 = hir.range_to_inclusive_expr %2
    ```
  }];

  let arguments = (ins AnyScalarObject:$right);
  let results = (outs AnyType:$output);

  //let assemblyFormat = "$output attr-dict  `:` functional-type(operands, results)";

  //let hasCanonicalizer = 1;
}

#endif
