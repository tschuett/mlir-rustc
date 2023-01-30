#ifndef HMirDialect
#define HMirDialect

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def HMir_Dialect : Dialect {
  let name = "hmir";
  let cppNamespace = "::rust_compiler::HMir";
  let description = [{
    Top-level dialect for HMir.

    This dialect tries to represent the HMIR dialect of not rustc.

  }];

  let extraClassDeclaration = [{

    static constexpr llvm::StringRef getBaseTypeAttrName() { return "BaseType"; }
    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }
  }];

  let useFoldAPI = kEmitFoldAdaptorFolder;
}

#endif
