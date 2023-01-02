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

def Mir_AddiOp : Mir_Op<"addi", [Pure]> {
  let summary = "Performs addition on the integer input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];
}

def AddfOp : Mir_Op<"addf", [Pure]> {
  let summary = "Performs addition on the float input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];
}


// CallOp

def CallOp : Mir_Op<"call", [CallOpInterface]> {
  let summary = "Generic call operation";
  let description = [{
    This operation represent a call to any function part of the sycl's namespace.
  }];

  let arguments = (ins
    OptionalAttr<FlatSymbolRefAttr>:$TypeName,
    FlatSymbolRefAttr:$FunctionName,
    FlatSymbolRefAttr:$MangledFunctionName,
    Variadic<AnyType>:$Args
  );

  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins
    "::llvm::Optional<::mlir::Type>":$result,
    "::llvm::Optional<::llvm::StringRef>":$TypeName,
    "::llvm::StringRef":$FunctionName,
    "::llvm::StringRef":$MangledFunctionName,
    "::mlir::ValueRange":$Args), [{
      odsState.addOperands(Args);
      if (TypeName.has_value()) {
        odsState.addAttribute(getTypeNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), TypeName.value()));
      }
      odsState.addAttribute(getFunctionNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), FunctionName));
      odsState.addAttribute(getMangledFunctionNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), MangledFunctionName));
      if (result.has_value()) {
        odsState.addTypes(result.value());
      }
    }]>
  ];

  let extraClassDeclaration = [{
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of the generic SYCL call operation, this is required by
    /// the call interface.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::FlatSymbolRefAttr>(getMangledFunctionNameAttrName());
    }

    /// Get the argument operands to the called function, this is required by the
    /// call interface.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
  }];

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

// HasParent<"mlir::func::FuncOp">
def ReturnOp : Mir_Op<"return", [Pure,
                                MemRefsNormalizable, ReturnLike, Terminator]> {
  let summary = "Function return operation";
  let description = [{
    The `Mir.return` operation represents a return operation within a function.
    The operation takes variable number of operands and produces no results.
    The operand number and types must match the signature of the function
    that contains the operation.

    Example:

    ```mlir
    Mir.func @foo() : (i32, f8) {
      ...
      return %0, %1 : i32, f8
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [OpBuilder<(ins), [{
    build($_builder, $_state, std::nullopt);
  }]>];

  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

def BorrowOp : Mir_Op<"borrow", [Pure]> {
  let summary = "Performs a borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

}

def MutBorrowOp : Mir_Op<"mutborrow", [Pure]> {
  let summary = "Performs a mutable borrow on the  input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];
}


def AddOp : Mir_Op<"add",
    [Pure]> {
  let summary = "addition operation";
  let description = [{
    The shapes of the  operands are expected to match.
  }];

//  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
//  let results = (outs F64Tensor);
//
//  // Indicate that the operation has a custom parser and printer method.
//  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def SubOp : Mir_Op<"sub",
    [Pure]> {
  let summary = "subtraction operation";
  let description = [{
    The shapes of the  operands are expected to match.
  }];

//  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
//  let results = (outs F64Tensor);
//
//  // Indicate that the operation has a custom parser and printer method.
//  let hasCustomAssemblyFormat = 1;

  // Allow building an SubOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}

def MulOp : Mir_Op<"mul",
    [Pure]> {
  let summary = "multiplication operation";
  let description = [{
      The shapes of the operands are expected to match.
  }];

//  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
//  let results = (outs F64Tensor);
//
//  // Indicate that the operation has a custom parser and printer method.
//  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "mlir::Value":$lhs, "mlir::Value":$rhs)>
  ];
}