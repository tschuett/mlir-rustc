#ifndef HirTypes
#define HirTypes


include "mlir/IR/AttrTypeBase.td"
include "HirDialect.td"


class Hir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Hir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Hir_StructType :
    DialectType<Hir_Dialect, CPred<"$_self.isa<StructType>()">,
                "Hir struct type">;

def Hir_EnumType :
    DialectType<Hir_Dialect, CPred<"$_self.isa<EnumType>()">,
                "Hir enum type">;

def Hir_StringType :
    DialectType<Hir_Dialect, CPred<"$_self.isa<StringType>()">,
                "Hir string type">;


#endif
