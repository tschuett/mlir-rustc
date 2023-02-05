#ifndef HirDialect
#define HirDialect

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Hir_Dialect : Dialect {
  let name = "hir";
  let cppNamespace = "::rust_compiler::hir";
  let description = [{
    Top-level dialect for Hir.

    This dialect tries to represent the HIR dialect of not rustc.

  }];

  let extraClassDeclaration = [{

    static constexpr llvm::StringRef getBaseTypeAttrName() { return "BaseType"; }
    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }
  }];

  let dependentDialects = [
    // Arith dialect provides Ops
    "arith::ArithDialect"
  ];

  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif
