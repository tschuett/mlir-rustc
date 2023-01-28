#ifndef MiRTypes
#define MirTypes

include "mlir/IR/AttrTypeBase.td"
include "MirDialect.td"


class Mir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Mir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

// Provide a definition for the Mir StructType for use in ODS. This allows for
// using StructType in a similar way to Tensor or MemRef. We use `DialectType`
// to demarcate the StructType as belonging to the Mir dialect.
def Mir_StructType :
    DialectType<Mir_Dialect, CPred<"$_self.isa<StructType>()">,
                "Mir struct type">;

#endif
