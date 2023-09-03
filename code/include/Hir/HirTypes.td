#ifndef HirTypes
#define HirTypes


include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/DataLayoutInterfaces.td"
include "HirDialect.td"
include "HirPattern.td"
include "HirOpsBase.td"

class Hir_Type<string name, string typeMnemonic, list<Trait> traits = [],
               string baseCppClass = "::mlir::Type">
    : TypeDef<Hir_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

//def Hir_ConstantType : Hir_Type<"Constant", "const"> {
//  let summary = "constant type";
//  let description = [{
//    constant.
//  }];
//  /// Here we defined a single parameter for the type, which is the bitwidth.
//  let parameters = (ins "unsigned":$width);
//
//  /// Here we define the textual format of the type declaratively, which will
//  /// automatically generate parser and printer logic. This will allow for
//  /// instances of the type to be output as, for example:
//  ///
//  ///    !my.int<10> // a 10-bit integer.
//  ///
//  let assemblyFormat = "`<` $width `>`";
//
//  /// Indicate that our type will add additional verification to the parameters.
//  //let genVerifyDecl = 1;
//}
//
//def Hir_EnumType :
//    DialectType<Hir_Dialect, CPred<"$_self.isa<EnumType>()">,
//                "Hir enum type">;
//
//def Hir_StringType :
//    DialectType<Hir_Dialect, CPred<"$_self.isa<StringType>()">,
//                "Hir string type">;

//===----------------------------------------------------------------------===//
// StructType
//
// The base type for all RecordDecls.
//
//===----------------------------------------------------------------------===//

def Hir_StructType : Hir_Type<"Struct", "struct",
    [DeclareTypeInterfaceMethods<DataLayoutTypeInterface>]> {

  let summary = "HIR struct type";
  let description = [{
    Each unique clang::RecordDecl is mapped to a `hir.struct` and any object in
    C/C++ that has a struct type will have a `hir.struct` in HIR.
  }];

  let parameters = (ins
    ArrayRefParameter<"mlir::Type", "members">:$members,
    "mlir::StringAttr":$typeName,
    "bool":$body,
    "bool":$packed
  );

  let builders = [
    TypeBuilder<(ins
      "llvm::ArrayRef<mlir::Type>":$members, "llvm::StringRef":$typeName,
      "bool":$body
    ), [{
      auto id = mlir::StringAttr::get(context, typeName);
      auto sTy = StructType::get(context, members, id, body,
                                 /*packed=*/false);
      return sTy;
    }]>
  ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
  private:
    // All these support lazily computation and storage
    // for the struct size and alignment.
    mutable std::optional<unsigned> size{}, align{};
    mutable std::optional<bool> padded{};
    void computeSizeAndAlignment(const ::mlir::DataLayout &dataLayout) const;
  public:
    void dropAst();
    size_t getNumElements() const { return getMembers().size(); }
    bool isOpaque() const { return !getBody(); }
    bool isPadded(const ::mlir::DataLayout &dataLayout) const;
  }];

  let extraClassDefinition = [{
    //void $cppClass::dropAst() {
    //  getImpl()->ast = std::nullopt;
    //}
  }];
}


#endif

