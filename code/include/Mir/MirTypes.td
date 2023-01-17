#ifndef MiRTypes
#define MirTypes

include "mlir/IR/AttrTypeBase.td"
include "MirDialect.td"


class Mir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Mir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

#endif