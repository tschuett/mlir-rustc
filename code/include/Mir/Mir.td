include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

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
    llvm::Optional<llvm::StringRef> findMethod(::mlir::TypeID Type,
                                               llvm::StringRef Name) const;

    /// Returns the name of the operation implementing the queried
    /// method, if present.
    ///
    /// For a method to be queried, it must have been registered
    /// first.
    ///
    /// This function is similar to findMethod(), but also works for
    /// methods registered as members of classes derived from \p BaseType.
    llvm::Optional<llvm::StringRef>
    findMethodFromBaseClass(::mlir::TypeID BaseType,
                            llvm::StringRef Name) const;

    /// Add a definition for the given method to be used when lowering
    /// SYCLMethodOpInterface instances.
    void registerMethodDefinition(llvm::StringRef Name, mlir::func::FuncOp Func);

    /// Retrieve a function definition previously registered with
    /// registerMethodDefinition().
    llvm::Optional<mlir::func::FuncOp>
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

def Mir_AddiOp : Mir_Op<"Addi", []> {
  let summary = "Performs addition on the integer input.";

  let description = [{
Each entry in `output` is the mean of the corresponding size `ksize`
window in `value`. NoMemoryEffect
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

def FuncOp : Mir_Op<"func", [
    DeclareOpInterfaceMethods<CallableOpInterface>, FunctionOpInterface,
    IsolatedFromAbove, Symbol
  ]> {
   let summary = "user defined function operation";
   let description = [{
   }];


   let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type
//    OptionalAttr<DictArrayAttr>:$arg_attrs,
//    OptionalAttr<DictArrayAttr>:$res_attrs
   );

   let regions = (region AnyRegion:$body);

//   let builders = [OpBuilder<(ins
//     "StringRef":$name, "FunctionType":$type,
//     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
//    >];


   let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

//    /// Returns the argument types of this function.
//    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }
//
//    /// Returns the result types of this function.
//    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
//  let skipDefaultBuilders = 1;
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
      odsState.addAttribute(MirDialect::getMangledFunctionNameAttrName(odsState.name), ::mlir::SymbolRefAttr::get(odsBuilder.getContext(), MirDialect::MangledFunctionName));
      if (result.has_value()) {
        odsState.addTypes(result.value());
      }
    }]>
  ];
 let extraClassDeclaration = [{
    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of the generic Mir call operation, this is required by
    /// the call interface.
    mlir::CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<mlir::FlatSymbolRefAttr>(MirDialect::getMangledFunctionNameAttrName());
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


