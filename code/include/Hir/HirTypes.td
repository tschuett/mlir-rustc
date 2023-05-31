#ifndef HirTypes
#define HirTypes


include "mlir/IR/AttrTypeBase.td"
include "HirDialect.td"


class Hir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Hir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def Hir_ConstantType : Hir_Type<"Constant", "const"> {
  let summary = "constant type";
  let description = [{
    constant.
  }];
  /// Here we defined a single parameter for the type, which is the bitwidth.
  //let parameters = (ins "unsigned":$width);

  /// Here we define the textual format of the type declaratively, which will
  /// automatically generate parser and printer logic. This will allow for
  /// instances of the type to be output as, for example:
  ///
  ///    !my.int<10> // a 10-bit integer.
  ///
  //let assemblyFormat = "`<` $width `>`";

  /// Indicate that our type will add additional verification to the parameters.
  //let genVerifyDecl = 1;
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
