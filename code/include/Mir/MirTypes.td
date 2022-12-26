include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"
include "MirBase.td"


def Mir_StructType :
    DialectType<Mir_Dialect, CPred<"$_self.isa<StructType>()">,
                "Mir struct type">;

