#ifndef HirOps
#define HirOps

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "HirDialect.td"
include "HirTypes.td"
include "HirAttr.td"
include "HirInterfaces.td"

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

def WhileLoopOp : Hir_Op<"while",
  []> {
  let summary = "a Rust while op.";

  let description = [{
  Rust PredicateLoopExpressionn
  }];

  let arguments = (ins AnyType:$rhs);

  let regions = (region AnyRegion:$region);
}

def InfiniteLoopOp : Hir_Op<"loop",
  []> {
  let summary = "a Rust loop op.";

  let description = [{
  Rust InfiniteLoopExpressionn
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

def BorrowOp : Hir_Op<"borrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins AnyType:$rhs);

  let results = (outs AnyType:$output);
}

def MutBorrowOp : Hir_Op<"mutborrow", [Pure, BorrowOpInterface]> {
  let summary = "Performs a mutable borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);

  let results = (outs AnyType:$output);
}

#endif
