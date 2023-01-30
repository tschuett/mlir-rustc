#ifndef HMirAttr
#define HMirAttr

include "HMirDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class HMirDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<HMir_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}


#endif
