include "mlir/IR/OpBase.td"

def Mir_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::mlir::mir::Mir";
  let description = [{
    Top-level dialect for interfacing PyTorch and MLIR.

    This dialect maintains a fairly isomorphic representation with TorchScript.

    This dialect also provides transforms that lower it to the
    "Torch backend contract", which is an IR form that we present to
    later conversions.
    The Torch backend contract significantly simplifies the IR representation
    and puts it in a form easier for later lowering to work on. Specifically:
    - The TorchScript object graph has been flattened to a list of globals (see
      the GlobalizeObjectGraph tranformation).
    - Most of the operations have been changed to operate on value-semantic
      tensors (see MaximizeValueSemantics)
    - The number of op variants have been reduced (see ReduceOpVariants)
    - Tensor sizes have been analyzed and static ranks inferred where possible
      and propagated throughout the program.
  }];

  let hasRegionArgAttrVerify = 1;
  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 0;

  let extraClassDeclaration = [{
    /// Parse a type registered to this dialect.
    Type parseType(DialectAsmParser &parser) const override;
    /// Print a type registered to this dialect.
    void printType(Type type, DialectAsmPrinter &printer) const override;
  }];
}
