set(LLVM_TARGET_DEFINITIONS HirAttr.td)
mlir_tablegen(HirEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(HirEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(HirAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(HirAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS HirOps.td)
mlir_tablegen(HirOps.h.inc -gen-op-decls)
mlir_tablegen(HirOps.cpp.inc -gen-op-defs)
mlir_tablegen(HirTypes.h.inc --gen-typedef-decls)
mlir_tablegen(HirTypes.cpp.inc --gen-typedef-defs)
mlir_tablegen(HirDialect.h.inc -gen-dialect-decls --dialect=hir)
mlir_tablegen(HirDialect.cpp.inc -gen-dialect-defs --dialect=hir)


set(LLVM_TARGET_DEFINITIONS HirOpsInterfaces.td)
mlir_tablegen(HirOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(HirOpsInterfaces.cpp.inc -gen-op-interface-defs)
#add_public_tablegen_target(HirOpsIncIFaceGen)

add_public_tablegen_target(HirOpsIncGen)
