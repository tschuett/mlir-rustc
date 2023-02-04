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


#endif
