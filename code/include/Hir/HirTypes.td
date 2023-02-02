#ifndef HirTypes
#define HirTypes


include "mlir/IR/AttrTypeBase.td"
include "HirDialect.td"


class Hir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Hir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}


#endif
