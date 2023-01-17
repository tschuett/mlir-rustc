#ifndef MirDialect
#define MirDialect

include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def Mir_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::rust_compiler::Mir";
  let description = [{
    Top-level dialect for Mir.

    This dialect tries to represent the MIR dialect of rustc.

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
