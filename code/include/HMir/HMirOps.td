#ifndef HMirOps
#define HMirOps


include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"
include "HMirDialect.td"
include "HMirTypes.td"
include "HMirAttr.td"
include "mlir/IR/BuiltinAttributes.td"
include "HMirInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

// Ops
class HMir_Op<string mnemonic, list<Trait> traits = []> :
    Op<HMir_Dialect, mnemonic, traits>;


def ErrorPropagationOp : HMir_Op<"question_mark", []> {
  let summary = "a Rust ? op.";

  let description = [{
  Rust ErrorPropagation Expression
  }];

  let arguments = (ins AnyType:$rhs);

  let regions = (region AnyRegion:$region);
}


def WhileLoopOp : HMir_Op<"while",
  [DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "a Rust while op.";

  let description = [{
  Rust PredicateLoopExpressionn
  }];

  let arguments = (ins AnyType:$rhs);

  let regions = (region AnyRegion:$region);
}

def IfOp : HMir_Op<"if", [NoRegionArguments]> {
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