#ifndef MirAttr
#define MirAttr

include "MirDialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"

class MirDialect_Attr<string name, string attrMnemonic, list<Trait> traits = []>
    : AttrDef<Mir_Dialect, name, traits> {
  let mnemonic = attrMnemonic;
}

def My_RustTypeAttr : MirDialect_Attr<"RustType", "type"> {
//  let parameters = (ins AttributeSelfTypeParameter<"">:$type, "APInt":$value);
}

#endif
