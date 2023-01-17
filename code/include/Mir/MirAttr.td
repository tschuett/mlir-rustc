#ifndef MirAttr
#define MirAttr

include "MirDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class MyDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Mir_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

#endif
