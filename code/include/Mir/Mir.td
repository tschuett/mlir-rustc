include "mlir/IR/OpBase.td"

def Mir_Dialect : Dialect {
  let name = "mir";
  let cppNamespace = "::mlir::mir::Mir";
  let description = [{
    Top-level dialect for Mir.

    This dialect tries to reprent the MIR dialect of rustc.

  }];

}
