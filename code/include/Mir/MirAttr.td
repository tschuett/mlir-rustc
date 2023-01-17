#ifndef MirAttr
#define MirAttr

include "MirDialect.td"
include "mlir/IR/EnumAttr.td"

class Mir_Attr<string name> : AttrDef<Mir_Dialect, name>;

#endif
