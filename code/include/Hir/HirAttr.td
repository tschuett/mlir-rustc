#ifndef HirAttr
#define HirAttr

include "HirDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class HirDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Hir_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}



#endif
