include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

def Mir_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::mlir::mir::Mir";
  let description = [{
    Top-level dialect for Mir.

    This dialect tries to reprsent the MIR dialect of rustc.

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

def Mir_AddfOp : Mir_Op<"Addi", []> {
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
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
   );

   let regions = (region AnyRegion:$body);

   let builders = [OpBuilder<(ins
     "StringRef":$name, "FunctionType":$type,
     CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
    >];


   let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];

  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}


// CallOp

def GenericCallOp : Mir_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<Mir_Type>:$inputs);

  let results = (outs Mir_Type);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}
