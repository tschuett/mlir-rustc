#add_mlir_dialect(MirOps mir)
#add_mlir_doc(Mir Mir Dialects/ -gen-dialect-doc)



set(LLVM_TARGET_DEFINITIONS MirAttr.td)
mlir_tablegen(MirEnumAttr.h.inc -gen-enum-decls)
mlir_tablegen(MirEnumAttr.cpp.inc -gen-enum-defs)
mlir_tablegen(MirAttr.h.inc --gen-attrdef-decls)
mlir_tablegen(MirAttr.cpp.inc -gen-attrdef-defs)

set(LLVM_TARGET_DEFINITIONS MirOps.td)
mlir_tablegen(MirOps.h.inc -gen-op-decls)
mlir_tablegen(MirOps.cpp.inc -gen-op-defs)
mlir_tablegen(MirTypes.h.inc --gen-typedef-decls)
mlir_tablegen(MirTypes.cpp.inc --gen-typedef-defs)
add_public_tablegen_target(MirOpsIncGen)


set(LLVM_TARGET_DEFINITIONS MirOpsInterfaces.td)
mlir_tablegen(MirOpsInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(MirOpsInterfaces.cpp.inc -gen-op-interface-defs)
