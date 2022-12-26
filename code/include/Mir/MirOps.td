include "MirTypes.td"

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
   let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }
  }];
}