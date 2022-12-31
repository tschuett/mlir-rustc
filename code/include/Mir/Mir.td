include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

include "MirOpsInterfaces.td"

def Mir_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::rust_compiler::Mir";
  let description = [{
    Top-level dialect for Mir.

    This dialect tries to represent the MIR dialect of rustc.

  }];

  let extraClassDeclaration = [{
    MethodRegistry methods;

    static constexpr llvm::StringRef getBaseTypeAttrName() { return "BaseType"; }
    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }

    /// Each operation will be registered as usual in an MLIR dialect
    /// and also will be registered as a SYCL method if it implements one.
    template <typename... Args> void addOperations();

    /// Returns the name of the operation implementing the queried
    /// method, if present.
    ///
    /// For a method to be queried, it must have been registered
    /// first.
    std::optional<llvm::StringRef> findMethod(::mlir::TypeID Type,
                                               llvm::StringRef Name) const;

    /// Returns the name of the operation implementing the queried
    /// method, if present.
    ///
    /// For a method to be queried, it must have been registered
    /// first.
    ///
    /// This function is similar to findMethod(), but also works for
    /// methods registered as members of classes derived from \p BaseType.
    std::optional<llvm::StringRef>
    findMethodFromBaseClass(::mlir::TypeID BaseType,
                            llvm::StringRef Name) const;

    /// Add a definition for the given method to be used when lowering
    /// SYCLMethodOpInterface instances.
    void registerMethodDefinition(llvm::StringRef Name, mlir::func::FuncOp Func);

    /// Retrieve a function definition previously registered with
    /// registerMethodDefinition().
    std::optional<mlir::func::FuncOp>
    lookupMethodDefinition(llvm::StringRef Name,
                           mlir::FunctionType Type) const;
  }];
}

// Types

def Mir_StructType :
    DialectType<Mir_Dialect, CPred<"$_self.isa<StructType>()">,
                "Mir struct type">;



// Ops
class Mir_Op<string mnemonic, list<Trait> traits = []> :
    Op<Mir_Dialect, mnemonic, traits>;


class MirMethodOpInterfaceImpl<
    string mnemonic, string type, list<string> methodNames, list<Trait> traits = []>
        : Mir_Op<mnemonic, !listconcat(traits, [MirMethodOpInterface, CallOpInterface])> {
  string baseType = type;
  list<string> memberFunctionNames = methodNames;
  int arrSize = !size(memberFunctionNames);

  let extraClassDeclaration = [{
    static constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> methods{{ }] #
      !interleave(!foreach(name, memberFunctionNames, "\"" # name # "\""), ", ") # [{
     }};
    static ::mlir::TypeID getTypeID() { return ::mlir::sycl::}] # type # [{::getTypeID(); }
    static constexpr llvm::ArrayRef<llvm::StringLiteral> getMethodNames() { return methods; }

    Operation::operand_iterator arg_operand_begin() { return (*this)->operand_begin(); }
    Operation::operand_iterator arg_operand_end() { return (*this)->operand_end(); }

    /// Return the callee of the generic Mir call operation, this is required by
    /// the call interface.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<FlatSymbolRefAttr>(getMangledFunctionNameAttrName());
    }

    /// Get the argument operands to the called function, this is required by the
    /// call interface.
    Operation::operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }
  }];

  let extraClassDefinition = [{
    constexpr std::array<llvm::StringLiteral, }] # arrSize # [{> $cppClass::methods;
  }];
}


def Mir_AddiOp : Mir_Op<"Addi", []> {
  let summary = "Performs addition on the integer input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];

  let extraClassDeclaration = [{
   static ::mlir::TypeID getTypeID() { return ::Mir::}] # type # [{::getTypeID(); }

   static constexpr llvm::ArrayRef<llvm::StringLiteral> getMethodNames() { return methods; }
  }];

  let builders = [];

  let hasVerifier = 1;
}

def AddfOp : Mir_Op<"Addi", []> {
  let summary = "Performs addition on the float input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
  }];


  let builders = [];

  let hasVerifier = 1;
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

  let assemblyFormat = [{
    `(` $Args `)` attr-dict `:` functional-type($Args, results)
  }];
}

def ReturnOp : Mir_Op<"return", [Pure, HasParent<"mlir::func::FuncOp">,
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
  let hasVerifier = 1;
}


