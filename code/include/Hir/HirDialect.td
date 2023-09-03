#ifndef HirDialect
#define HirDialect

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

def Hir_Dialect : Dialect {
  let name = "hir";
  let cppNamespace = "::rust_compiler::hir";
  let description = [{
    Top-level dialect for Hir.

    This dialect tries to represent the HIR dialect of not rustc.
    It tries to closely model the source language.
  }];

  let extraClassDeclaration = [{
    void registerTypes();

    static constexpr llvm::StringRef getBaseTypeAttrName() { return "BaseType"; }
    static constexpr llvm::StringRef getFunctionNameAttrName() { return "FunctionName"; }
    static constexpr llvm::StringRef getMangledFunctionNameAttrName() { return "MangledFunctionName"; }
    static constexpr llvm::StringRef getTypeNameAttrName() { return "TypeName"; }

  }];

  //let hasConstantMaterializer = true;

  let dependentDialects = [
                           "mlir::cf::ControlFlowDialect",
                           "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::vector::VectorDialect"
                          ];


   let useDefaultTypePrinterParser = true;
   let useDefaultAttributePrinterParser = true;
   let usePropertiesForAttributes = 1;
}

#endif
