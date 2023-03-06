#ifndef LirDialect
#define LirDialect

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Lir_Dialect : Dialect {
  let name = "lir";
  let cppNamespace = "::rust_compiler::Lir";
  let description = [{
    Bottom-level dialect for Lir.

    This dialect tries to represent the LIR dialect of rustc.

  }];

  let extraClassDeclaration = [{

    static constexpr llvm::StringRef getBaseTypeAttrName() { return "BaseType"; }
    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }
  }];
}

#endif
