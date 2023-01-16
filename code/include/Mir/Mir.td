include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"

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

// Types

////////////////////////////////////////////////////////////////////////////////
// TYPES DECLARATIONS
////////////////////////////////////////////////////////////////////////////////

class Mir_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Mir_Dialect, name,
              !listconcat([MemRefElementTypeInterface,
                           LLVM_PointerElementTypeInterface],
                               traits)> {
  let mnemonic = typeMnemonic;
}


// Ops
class Mir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mir_Dialect, mnemonic, traits>;

def BorrowOp : Mir_Op<"borrow", [Pure]> {
  let summary = "Performs a borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`.
  }];

  let arguments = (ins AnyType:$rhs);
}

def MutBorrowOp : Mir_Op<"mutborrow", [Pure]> {
  let summary = "Performs a mutable borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);
}

def AwaitOp : Mir_Op<"await", [Pure]> {
  let summary = "a Rust await op.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let arguments = (ins AnyType:$rhs);
}
